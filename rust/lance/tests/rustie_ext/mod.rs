// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The extension surface on a real dataset: stable row ids, a registered tokenizer, appends that
//! become further index segments, deletes, and compaction. Reads the FTS index through the
//! partition / cursor / row-id accessors and checks it against the table itself.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};
use lance_index::scalar::inverted::{
    InvertedIndex, InvertedIndexParams, Language, TERMINATED, register_tokenizer, unregister_tokenizer,
};
use lance_tokenizer::{BoxTokenStream, Token, TokenStream};

const TOKENIZER: &str = "rustie-test/slots-dataset";

/// One term per `,`-separated entry of a whitespace-separated slot; a slot's terms share its
/// index as their position; `_` is an empty slot that still advances the position.
#[derive(Debug, Clone)]
struct SlotTokenizer;

struct SlotStream {
    tokens: Vec<Token>,
    next: usize,
}

impl TokenStream for SlotStream {
    fn advance(&mut self) -> bool {
        self.next += 1;
        self.next <= self.tokens.len()
    }

    fn token(&self) -> &Token {
        &self.tokens[self.next - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.next - 1]
    }
}

fn slot_stream<'a>(text: &str) -> BoxTokenStream<'a> {
    let mut tokens = Vec::new();
    for (position, slot) in text.split_whitespace().enumerate() {
        if slot == "_" {
            continue;
        }
        for term in slot.split(',') {
            tokens.push(Token {
                offset_from: 0,
                offset_to: term.len(),
                position,
                text: term.to_string(),
                position_length: 1,
            });
        }
    }
    BoxTokenStream::new(SlotStream { tokens, next: 0 })
}

impl LanceTokenizer for SlotTokenizer {
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a> {
        slot_stream(query_text)
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        slot_stream(text)
    }

    fn box_clone(&self) -> Box<dyn LanceTokenizer> {
        Box::new(self.clone())
    }

    fn doc_type(&self) -> DocType {
        DocType::Text
    }
}

/// `common` at positions 0 and 3, a document-specific term at 0, `third` at 1 for every third id.
fn slots_of(id: i32) -> String {
    let third = if id % 3 == 0 { "third" } else { "_" };
    format!("common,d{} {third} _ common", id % 5)
}

fn batch(ids: std::ops::Range<i32>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("slots", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(StringArray::from_iter_values(ids.map(slots_of))),
        ],
    )
    .unwrap()
}

/// `(_rowid -> id)` of every live row, from the table.
async fn live_rows(ds: &Dataset) -> BTreeMap<u64, i32> {
    let data = ds.scan().with_row_id().project(&["id"]).unwrap().try_into_batch().await.unwrap();
    let ids = data["id"].as_primitive::<Int32Type>();
    let rows = data["_rowid"].as_primitive::<UInt64Type>();
    rows.values().iter().copied().zip(ids.values().iter().copied()).collect()
}

/// For one token: `(row id, positions)` over every partition of every segment of the index.
async fn token_postings(ds: &Dataset, token: &str) -> BTreeMap<u64, Vec<u32>> {
    let mut out = BTreeMap::new();
    let metrics = NoOpMetricsCollector;
    for segment in ds.load_indices_by_name("slots_idx").await.unwrap() {
        let index = ds.open_scalar_index("slots", &segment.uuid, &metrics).await.unwrap();
        let index = index.as_any().downcast_ref::<InvertedIndex>().expect("an FTS index");
        for partition in index.partitions() {
            let Some(token_id) = partition.token_id(token) else { continue };
            let rows = partition.rows().await.unwrap();
            let mut cursor = partition.posting_cursor(token_id, true, &metrics).await.unwrap();
            let mut positions = Vec::new();
            while cursor.doc() != TERMINATED {
                cursor.positions(&mut positions).unwrap();
                if let Some(row) = rows.row_id(cursor.doc()) {
                    assert!(out.insert(row, positions.clone()).is_none(), "row {row} indexed twice");
                }
                cursor.advance();
            }
        }
    }
    out
}

#[tokio::test]
async fn test_extension_index_follows_stable_row_ids_across_append_delete_and_compaction() {
    register_tokenizer(TOKENIZER, Arc::new(|_| Ok(Box::new(SlotTokenizer) as Box<dyn LanceTokenizer>))).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();

    // 900 rows in three fragments, with stable row ids.
    let params = WriteParams {
        mode: WriteMode::Create,
        enable_stable_row_ids: true,
        max_rows_per_file: 300,
        ..Default::default()
    };
    let first = batch(0..900);
    let mut ds = Dataset::write(RecordBatchIterator::new(vec![Ok(first.clone())], first.schema()), uri, Some(params))
        .await
        .unwrap();
    assert_eq!(ds.get_fragments().len(), 3);
    ds.create_index(
        &["slots"],
        IndexType::Inverted,
        Some("slots_idx".to_string()),
        &InvertedIndexParams::new(TOKENIZER.to_string(), Language::English).with_position(true),
        true,
    )
    .await
    .unwrap();

    // The index agrees with the table: one posting per row for `common`, with positions 0 and 3,
    // and the row ids it holds are the table's `_rowid`s (stable ids, not addresses).
    let rows = live_rows(&ds).await;
    assert_eq!(rows.len(), 900);
    let common = token_postings(&ds, "common").await;
    assert_eq!(common.keys().copied().collect::<BTreeSet<_>>(), rows.keys().copied().collect::<BTreeSet<_>>());
    assert!(common.values().all(|positions| positions == &[0, 3]));
    let third = token_postings(&ds, "third").await;
    let thirds: BTreeSet<u64> = rows.iter().filter(|(_, id)| *id % 3 == 0).map(|(row, _)| *row).collect();
    assert_eq!(third.keys().copied().collect::<BTreeSet<_>>(), thirds);
    assert!(third.values().all(|positions| positions == &[1]));

    // An append is a new fragment the index does not cover until optimize adds a segment.
    let more = batch(900..1200);
    ds.append(RecordBatchIterator::new(vec![Ok(more.clone())], more.schema()), None).await.unwrap();
    assert_eq!(token_postings(&ds, "common").await.len(), 900, "unindexed rows are invisible to the index");
    ds.optimize_indices(&OptimizeOptions::append()).await.unwrap();
    assert_eq!(ds.load_indices_by_name("slots_idx").await.unwrap().len(), 2, "a second segment");
    let rows = live_rows(&ds).await;
    assert_eq!(rows.len(), 1200);
    let common = token_postings(&ds, "common").await;
    assert_eq!(common.keys().copied().collect::<BTreeSet<_>>(), rows.keys().copied().collect::<BTreeSet<_>>());

    // Deleted rows stay in the index (the dataset's deletion mask applies at query time); compaction
    // rewrites fragments, and with stable row ids the surviving rows keep their ids.
    ds.delete("id % 10 = 0").await.unwrap();
    let before_compaction = live_rows(&ds).await;
    assert_eq!(before_compaction.len(), 1200 - 120);
    compact_files(&mut ds, CompactionOptions { target_rows_per_fragment: 10_000, ..Default::default() }, None)
        .await
        .unwrap();
    assert!(ds.get_fragments().len() < 4, "compaction merged fragments");
    let after_compaction = live_rows(&ds).await;
    assert_eq!(after_compaction, before_compaction, "compaction keeps stable row ids");
    let common = token_postings(&ds, "common").await;
    let live: BTreeSet<u64> = after_compaction.keys().copied().collect();
    let indexed: BTreeSet<u64> = common.keys().copied().collect();
    assert!(live.is_subset(&indexed), "every live row is still indexed under its row id");
    assert!(common.values().all(|positions| positions == &[0, 3]));

    // Row ids from the index fetch the right rows: the stored text tokenizes to what the index holds.
    let sample: Vec<u64> = live.iter().copied().step_by(97).collect();
    let fetched = ds.take_rows(&sample, ds.schema().clone()).await.unwrap();
    let ids = fetched["id"].as_primitive::<Int32Type>();
    let by_row: HashMap<u64, i32> = after_compaction.iter().map(|(r, i)| (*r, *i)).collect();
    assert_eq!(ids.len(), sample.len());
    for (row, id) in sample.iter().zip(ids.values()) {
        assert_eq!(by_row[row], *id);
    }

    assert!(unregister_tokenizer(TOKENIZER));
}
