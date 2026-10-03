// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests of the extension surface: a registered tokenizer with explicit positions, an
//! index built and reloaded with it, and the partition / cursor / row-id accessors read back
//! against a model of what was indexed.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempDir;
use lance_io::object_store::ObjectStore;
use lance_tokenizer::{BoxTokenStream, Token, TokenStream};

use crate::metrics::NoOpMetricsCollector;
use crate::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};
use crate::scalar::inverted::{
    InvertedIndex, InvertedIndexBuilder, InvertedIndexParams, Language, PostingCursor, TERMINATED,
    register_tokenizer, unregister_tokenizer,
};
use crate::scalar::lance_format::LanceIndexStore;

/// One term per `,`-separated entry of a whitespace-separated slot; the terms of a slot share the
/// slot's index as their position, and a `_` slot emits nothing but still takes an index.
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

impl SlotTokenizer {
    fn stream<'a>(text: &str) -> BoxTokenStream<'a> {
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
}

impl LanceTokenizer for SlotTokenizer {
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a> {
        Self::stream(query_text)
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        Self::stream(text)
    }

    fn box_clone(&self) -> Box<dyn LanceTokenizer> {
        Box::new(self.clone())
    }

    fn doc_type(&self) -> DocType {
        DocType::Text
    }
}

const ROW_BASE: u64 = 10_000;

/// Document `i`'s slots, and the terms with their positions that they must produce.
fn document(i: usize) -> (String, Vec<(String, u32)>) {
    let mut slots = Vec::new();
    let mut terms: Vec<(String, u32)> = Vec::new();
    // Position 0: two terms; every document has `common`.
    slots.push(format!("common,m{}", i % 7));
    terms.push(("common".to_string(), 0));
    terms.push((format!("m{}", i % 7), 0));
    // Position 1: only every third document.
    if i.is_multiple_of(3) {
        slots.push("third".to_string());
        terms.push(("third".to_string(), 1));
    } else {
        slots.push("_".to_string());
    }
    // Position 2 is always empty: a gap.
    slots.push("_".to_string());
    // Position 3: `common` again (frequency 2), and `eleven` for every eleventh document.
    if i.is_multiple_of(11) {
        slots.push("common,eleven".to_string());
        terms.push(("eleven".to_string(), 3));
    } else {
        slots.push("common".to_string());
    }
    terms.push(("common".to_string(), 3));
    (slots.join(" "), terms)
}

fn batch(range: std::ops::Range<usize>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("slots", DataType::Utf8, true),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));
    let docs: Vec<String> = range.clone().map(|i| document(i).0).collect();
    let rows: Vec<u64> = range.map(|i| ROW_BASE + i as u64).collect();
    RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(docs)), Arc::new(UInt64Array::from(rows))],
    )
    .unwrap()
}

async fn build_index(name: &str, num_docs: usize, with_position: bool) -> Arc<InvertedIndex> {
    register_tokenizer(name, Arc::new(|_| Ok(Box::new(SlotTokenizer) as Box<dyn LanceTokenizer>))).unwrap();
    let dir = TempDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        dir.obj_path(),
        Arc::new(LanceCache::no_cache()),
    ));
    let params = InvertedIndexParams::new(name.to_string(), Language::English).with_position(with_position);
    let batches: Vec<_> = (0..num_docs).step_by(400).map(|s| Ok(batch(s..(s + 400).min(num_docs)))).collect();
    let schema = batch(0..1).schema();
    let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
    InvertedIndexBuilder::new(params)
        .update(Box::pin(stream), store.as_ref(), None)
        .await
        .unwrap();
    // The directory must outlive the index: it is dropped with the test's store.
    std::mem::forget(dir);
    InvertedIndex::load(store, None, &LanceCache::no_cache()).await.unwrap()
}

/// `term -> [(doc index, positions)]`, from the generator.
fn model(num_docs: usize) -> BTreeMap<String, Vec<(usize, Vec<u32>)>> {
    let mut out: BTreeMap<String, Vec<(usize, Vec<u32>)>> = BTreeMap::new();
    for i in 0..num_docs {
        let mut per_term: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (term, position) in document(i).1 {
            per_term.entry(term).or_default().push(position);
        }
        for (term, mut positions) in per_term {
            positions.sort_unstable();
            out.entry(term).or_default().push((i, positions));
        }
    }
    out
}

fn drain(cursor: &mut PostingCursor, with_positions: bool) -> Vec<(u32, u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut positions = Vec::new();
    while cursor.doc() != TERMINATED {
        if with_positions {
            cursor.positions(&mut positions).unwrap();
        }
        out.push((cursor.doc(), cursor.freq(), positions.clone()));
        cursor.advance();
    }
    out
}

#[tokio::test]
async fn test_cursor_reads_what_a_custom_tokenizer_indexed() {
    const NUM_DOCS: usize = 1_000; // 7 full blocks and a tail for `common`, partial blocks elsewhere
    let name = "rustie-test/slots-e2e";
    let index = build_index(name, NUM_DOCS, true).await;
    let expected = model(NUM_DOCS);
    let metrics = NoOpMetricsCollector;

    let mut checked_terms = 0;
    let mut docs_seen = 0usize;
    for partition in index.partitions() {
        assert!(!partition.is_legacy());
        let rows = partition.rows().await.unwrap();
        let fst = partition.token_fst().expect("a loaded partition has an FST dictionary");
        let mut terms = Vec::new();
        {
            let mut s = fst::IntoStreamer::into_stream(fst);
            while let Some((token, id)) = fst::Streamer::next(&mut s) {
                terms.push((String::from_utf8(token.to_vec()).unwrap(), id as u32));
            }
        }
        for (term, token_id) in terms {
            assert_eq!(partition.token_id(&term), Some(token_id));
            let df = partition.doc_freq(token_id, None).await.unwrap();
            let mut cursor = partition.posting_cursor(token_id, true, &metrics).await.unwrap();
            assert_eq!(cursor.len() as u32, df, "doc_freq of {term}");
            let got = drain(&mut cursor, true);
            assert_eq!(got.len() as u32, df);
            let want = &expected[&term];
            // Doc ids are partition-local; the row id says which document it is.
            let mut by_doc: Vec<(usize, u32, Vec<u32>)> = got
                .iter()
                .map(|(doc, freq, positions)| {
                    let row = rows.row_id(*doc).expect("no dead slots here");
                    ((row - ROW_BASE) as usize, *freq, positions.clone())
                })
                .collect();
            by_doc.sort();
            let want_docs: Vec<(usize, u32, Vec<u32>)> =
                want.iter().map(|(i, p)| (*i, p.len() as u32, p.clone())).collect();
            assert_eq!(by_doc, want_docs, "postings of {term}");
            checked_terms += 1;
            docs_seen += got.len();
        }
    }
    assert_eq!(checked_terms, expected.len(), "every term of the model is in some dictionary");
    assert!(docs_seen > NUM_DOCS);
    assert!(unregister_tokenizer(name));
}

#[tokio::test]
async fn test_cursor_seek_agrees_with_a_model() {
    const NUM_DOCS: usize = 1_000;
    let name = "rustie-test/slots-seek";
    let index = build_index(name, NUM_DOCS, true).await;
    let metrics = NoOpMetricsCollector;
    let partition = &index.partitions()[0];

    for term in ["common", "m3", "third", "eleven"] {
        let token_id = partition.token_id(term).unwrap();
        let mut all = partition.posting_cursor(token_id, false, &metrics).await.unwrap();
        let docs: Vec<u32> = drain(&mut all, false).into_iter().map(|(doc, ..)| doc).collect();
        assert!(docs.windows(2).all(|w| w[0] < w[1]), "doc ids ascend");
        let max_doc = *docs.last().unwrap();

        // Ascending seek targets from a small LCG, including repeats and targets past the end.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut targets: Vec<u32> = (0..300)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as u32) % (max_doc + 40)
            })
            .collect();
        targets.sort_unstable();
        let mut cursor = partition.posting_cursor(token_id, true, &metrics).await.unwrap();
        for target in targets {
            let want = docs.iter().copied().find(|&d| d >= target).unwrap_or(TERMINATED);
            let current = cursor.doc();
            let got = cursor.seek(target);
            // A cursor never moves backwards: a target at or below the current doc stays put.
            let want = if current != TERMINATED && current >= target { current } else { want };
            assert_eq!(got, want, "seek({target}) on {term}");
            assert_eq!(cursor.doc(), got);
            if got != TERMINATED {
                let mut positions = Vec::new();
                cursor.positions(&mut positions).unwrap();
                assert_eq!(positions.len() as u32, cursor.freq());
            }
        }
        // Seeking past everything exhausts the cursor, and it stays exhausted.
        assert_eq!(cursor.seek(u32::MAX - 1), TERMINATED);
        assert_eq!(cursor.advance(), TERMINATED);
        assert_eq!(cursor.seek(0), TERMINATED);
    }
    assert!(unregister_tokenizer(name));
}

#[tokio::test]
async fn test_positions_require_an_index_built_with_positions() {
    let name = "rustie-test/slots-nopos";
    let index = build_index(name, 200, false).await;
    let partition = &index.partitions()[0];
    let token_id = partition.token_id("common").unwrap();
    assert_eq!(partition.doc_freq(token_id, None).await.unwrap(), 200);
    assert!(partition.posting_cursor(token_id, true, &NoOpMetricsCollector).await.is_err());
    let mut cursor = partition.posting_cursor(token_id, false, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(drain(&mut cursor, false).len(), 200);
    let mut positions = Vec::new();
    assert!(cursor.positions(&mut positions).is_err(), "no positions were asked for");
    assert!(unregister_tokenizer(name));
}

#[tokio::test]
async fn test_an_index_opens_without_its_tokenizer_but_cannot_be_extended() {
    let name = "rustie-test/slots-lost";
    let index = build_index(name, 50, true).await;
    let params = index.params().clone();
    assert!(unregister_tokenizer(name));
    // Opening for reads works (the placeholder tokenizer has no tokens), and the cursor and
    // dictionary accessors do not need the tokenizer at all.
    let partition = &index.partitions()[0];
    let token_id = partition.token_id("common").unwrap();
    let mut cursor = partition.posting_cursor(token_id, true, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(drain(&mut cursor, true).len(), 50);
    // Tokenizing documents (create / update) fails loudly rather than indexing nothing.
    assert!(params.build().is_err());
}
