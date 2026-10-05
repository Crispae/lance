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
use lance_file::version::ConcreteFileVersion;
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
    // Position 4: many rare terms (a few documents each), so the dictionary has hundreds of tokens
    // with short, scattered postings; every fifth document has a second one at the same position.
    if i.is_multiple_of(5) {
        slots.push(format!("z{},y{}", i % 300, i % 40));
        terms.push((format!("z{}", i % 300), 4));
        terms.push((format!("y{}", i % 40), 4));
    } else {
        slots.push(format!("z{}", i % 300));
        terms.push((format!("z{}", i % 300), 4));
    }
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
    build_index_with_cache(name, num_docs, with_position, &LanceCache::no_cache()).await
}

/// The index holds its cache weakly: the caller must keep `cache` alive for as long as it uses the index.
async fn build_index_with_cache(name: &str, num_docs: usize, with_position: bool, cache: &LanceCache) -> Arc<InvertedIndex> {
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
    InvertedIndex::load(store, None, cache).await.unwrap()
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

/// Every term of the partition's dictionary, as token ids.
fn all_token_ids(partition: &crate::scalar::inverted::InvertedPartition) -> Vec<u32> {
    let fst = partition.token_fst().expect("a loaded partition has an FST dictionary");
    let mut ids = Vec::new();
    let mut s = fst::IntoStreamer::into_stream(fst);
    while let Some((_, id)) = fst::Streamer::next(&mut s) {
        ids.push(id as u32);
    }
    ids.sort_unstable();
    ids
}

/// The bulk path must give exactly the per-token path's postings and positions, for every token,
/// for scattered, repeated and unordered requests, with and without a cache (the second call is
/// served from it), and `doc_freq` must not change once the metadata is resident.
#[tokio::test]
async fn test_bulk_cursors_equal_per_token_cursors() {
    const NUM_DOCS: usize = 1_000;
    let metrics = NoOpMetricsCollector;
    for (label, cache) in [("uncached", LanceCache::no_cache()), ("cached", LanceCache::with_capacity(64 << 20))] {
        let name = format!("rustie-test/slots-bulk-{label}");
        let index = build_index_with_cache(&name, NUM_DOCS, true, &cache).await;
        for partition in index.partitions() {
            let ids = all_token_ids(partition);
            assert!(ids.len() > 50, "enough tokens to scatter requests ({})", ids.len());

            // Sorted, shuffled with repeats, a sparse subset, and a single token.
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            let mut shuffled: Vec<u32> = ids.iter().copied().chain(ids.iter().copied().take(20)).collect();
            for i in (1..shuffled.len()).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                shuffled.swap(i, (state >> 33) as usize % (i + 1));
            }
            let sparse: Vec<u32> = ids.iter().copied().step_by(7).collect();
            let requests = [ids.clone(), shuffled, sparse, vec![ids[ids.len() / 2]]];

            // First with no counts resident (only the `_length` rows needed are read), then with the
            // `_length` column alone, then with the full metadata (both let the bulk reads bridge the rows
            // between scattered tokens).
            for resident in ["none", "lengths", "metadata"] {
                match resident {
                    "lengths" => partition.load_term_lengths().await.unwrap(),
                    "metadata" => partition.load_term_metadata().await.unwrap(),
                    _ => {}
                }
                for with_positions in [true, false] {
                    for request in &requests {
                        for pass in 0..2 {
                            let mut bulk = partition.posting_cursors(request, with_positions, &metrics).await.unwrap();
                            assert_eq!(bulk.len(), request.len());
                            for (cursor, &token) in bulk.iter_mut().zip(request) {
                                let mut single = partition.posting_cursor(token, with_positions, &metrics).await.unwrap();
                                assert_eq!(cursor.len(), single.len(), "length of token {token}");
                                assert_eq!(
                                    drain(cursor, with_positions),
                                    drain(&mut single, with_positions),
                                    "token {token} ({label}, resident {resident}, positions {with_positions}, pass {pass})"
                                );
                            }
                        }
                    }
                }
                // Document counts, in the order asked (repeats included), equal the per-token lookups.
                let counts = partition.doc_freqs(&requests[1]).await.unwrap();
                for (&token, count) in requests[1].iter().zip(counts) {
                    assert_eq!(count, partition.doc_freq(token, None).await.unwrap(), "doc_freq of token {token} (resident {resident})");
                }
            }
        }
        assert!(unregister_tokenizer(&name));
    }
}

#[tokio::test]
async fn test_bulk_positions_require_an_index_built_with_positions() {
    let name = "rustie-test/slots-bulk-nopos";
    let index = build_index(name, 200, false).await;
    let partition = &index.partitions()[0];
    let ids = all_token_ids(partition);
    assert!(partition.posting_cursors(&ids, true, &NoOpMetricsCollector).await.is_err());
    let mut cursors = partition.posting_cursors(&ids, false, &NoOpMetricsCollector).await.unwrap();
    let total: usize = cursors.iter_mut().map(|cursor| drain(cursor, false).len()).sum();
    assert!(total >= 200);
    assert!(unregister_tokenizer(name));
}

/// Builds an index on disk and keeps its directory, so it can be loaded again (with its own object
/// store and cache) to count reads.
async fn build_files(name: &str, num_docs: usize) -> TempDir {
    build_files_in(name, num_docs, ConcreteFileVersion::V2_0).await
}

/// [`build_files`] writing the index files in file format `version`.
async fn build_files_in(name: &str, num_docs: usize, version: ConcreteFileVersion) -> TempDir {
    register_tokenizer(name, Arc::new(|_| Ok(Box::new(SlotTokenizer) as Box<dyn LanceTokenizer>))).unwrap();
    let dir = TempDir::default();
    let store = Arc::new(LanceIndexStore::with_format_version(Arc::new(ObjectStore::local()), dir.obj_path(), Arc::new(LanceCache::no_cache()), version));
    let params = InvertedIndexParams::new(name.to_string(), Language::English).with_position(true);
    let batches: Vec<_> = (0..num_docs).step_by(400).map(|s| Ok(batch(s..(s + 400).min(num_docs)))).collect();
    let schema = batch(0..1).schema();
    let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
    InvertedIndexBuilder::new(params).update(Box::pin(stream), store.as_ref(), None).await.unwrap();
    dir
}

/// Concurrent bulk reads of the same tokens must share one read per token: the cache alone does not
/// (every caller misses before the first one has inserted), so the object-store reads of eight
/// concurrent calls must equal those of one call.
#[tokio::test]
async fn test_concurrent_bulk_reads_share_one_read_per_token() {
    let name = "rustie-test/slots-bulk-concurrent";
    let dir = build_files(name, 1_000).await;
    let reads = |concurrency: usize| {
        let dir = &dir;
        async move {
            let object_store = Arc::new(ObjectStore::local());
            let store = Arc::new(LanceIndexStore::new(object_store.clone(), dir.obj_path(), Arc::new(LanceCache::no_cache())));
            // The index holds its cache weakly: keep it alive while the index is used.
            let cache = LanceCache::with_capacity(64 << 20);
            let index = InvertedIndex::load(store, None, &cache).await.unwrap();
            let partition = &index.partitions()[0];
            let ids = all_token_ids(partition);
            object_store.io_stats_incremental();
            let calls = (0..concurrency).map(|_| partition.posting_cursors(&ids, true, &NoOpMetricsCollector));
            for result in futures::future::join_all(calls).await {
                assert_eq!(result.unwrap().len(), ids.len());
            }
            object_store.io_stats_incremental().read_iops
        }
    };
    let one = reads(1).await;
    let eight = reads(8).await;
    assert!(one > 0, "the bulk read touched the object store");
    assert_eq!(eight, one, "eight concurrent calls must not re-read what one call reads");
    assert!(unregister_tokenizer(name));
}

/// Concurrent readers of different tokens still share one initialization of each cold page: the
/// cache alone does not share it, so before the fix every reader read the page metadata again.
/// Eight concurrent callers, each with its own token, must read no more than the same callers one
/// after another. Page initialization exists only in the structural formats (2.1+), as on S3.
#[tokio::test]
async fn test_concurrent_cold_readers_share_page_initialization() {
    let name = "rustie-test/slots-page-init";
    let dir = build_files_in(name, 60_000, ConcreteFileVersion::V2_2).await;
    let reads = |concurrent: bool| {
        let dir = &dir;
        async move {
            let object_store = Arc::new(ObjectStore::local());
            // Page metadata is cached in the store's (file) cache, as in a dataset session.
            let store = Arc::new(LanceIndexStore::new(object_store.clone(), dir.obj_path(), Arc::new(LanceCache::with_capacity(64 << 20))));
            let cache = LanceCache::with_capacity(64 << 20);
            let index = InvertedIndex::load(store, None, &cache).await.unwrap();
            let partition = &index.partitions()[0];
            let ids = all_token_ids(partition);
            // Eight tokens spread over the dictionary, one per caller.
            let picks: Vec<u32> = (0..8).map(|i| ids[i * (ids.len() - 1) / 7]).collect();
            object_store.io_stats_incremental();
            if concurrent {
                let calls = picks.iter().map(|id| partition.posting_cursors(std::slice::from_ref(id), true, &NoOpMetricsCollector));
                for result in futures::future::join_all(calls).await {
                    assert_eq!(result.unwrap().len(), 1);
                }
            } else {
                for id in &picks {
                    assert_eq!(partition.posting_cursors(std::slice::from_ref(id), true, &NoOpMetricsCollector).await.unwrap().len(), 1);
                }
            }
            object_store.io_stats_incremental().read_iops
        }
    };
    let sequential = reads(false).await;
    let concurrent = reads(true).await;
    assert!(sequential > 0, "the reads touched the object store");
    assert!(concurrent <= sequential, "concurrent cold readers read {concurrent} times, one after another {sequential}");
    assert!(unregister_tokenizer(name));
}

/// `rows()` loads the document row ids and lengths in one request batch, so opening a cold
/// partition costs fewer reads than loading the two columns one after the other (what
/// `address_keyed` alone does), and returns the same table. The files are small, as for a segment of
/// a few thousand sentences: below the store's block size each read fetches the whole object, so
/// every extra read is a full download.
#[tokio::test]
async fn test_rows_load_the_document_columns_together() {
    let name = "rustie-test/slots-rows-open";
    register_tokenizer(name, Arc::new(|_| Ok(Box::new(SlotTokenizer) as Box<dyn LanceTokenizer>))).unwrap();
    // A block size above the file sizes makes every read fetch the whole object, as on S3 for a small
    // segment (the default there is 64 KiB).
    let temp = TempDir::default();
    let params = lance_io::object_store::ObjectStoreParams { block_size: Some(1 << 20), ..Default::default() };
    let (object_store, dir) = ObjectStore::from_uri_and_params(Default::default(), &format!("file://{}", temp.path_str()), &params).await.unwrap();
    let build = LanceIndexStore::with_format_version(object_store.clone(), dir.clone(), Arc::new(LanceCache::no_cache()), ConcreteFileVersion::V2_2);
    let params = InvertedIndexParams::new(name.to_string(), Language::English).with_position(true);
    let batches: Vec<_> = (0..2_000).step_by(400).map(|s| Ok(batch(s..(s + 400).min(2_000)))).collect();
    let schema = batch(0..1).schema();
    let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
    InvertedIndexBuilder::new(params).update(Box::pin(stream), &build, None).await.unwrap();
    let open = || {
        let (object_store, dir) = (object_store.clone(), dir.clone());
        async move {
            let store = Arc::new(LanceIndexStore::with_format_version(object_store.clone(), dir, Arc::new(LanceCache::with_capacity(64 << 20)), ConcreteFileVersion::V2_2));
            let cache = LanceCache::with_capacity(64 << 20);
            let index = InvertedIndex::load(store, None, &cache).await.unwrap();
            object_store.io_stats_incremental();
            (index, cache)
        }
    };
    let (index, _cache) = open().await;
    let rows = index.partitions()[0].rows().await.unwrap();
    let together = object_store.io_stats_incremental().read_iops;
    let (index, _cache) = open().await;
    let separate_rows = index.partitions()[0].docs.address_keyed().await.unwrap();
    let separate = object_store.io_stats_incremental().read_iops;
    assert!(together < separate, "together {together} reads, separately {separate}");
    assert_eq!(rows.len(), 2_000);
    assert_eq!(separate_rows.len(), 2_000);
    for doc in [0u32, 1, 999, 1_999] {
        assert_eq!(rows.row_id(doc), index.partitions()[0].rows().await.unwrap().row_id(doc));
    }
    assert!(unregister_tokenizer(name));
}

/// `load_term_lengths` reads the `_length` column alone: fewer bytes than the full metadata (which adds
/// `_max_score`), and afterwards document counts cost no reads.
#[tokio::test]
async fn test_term_lengths_load_without_max_scores() {
    let name = "rustie-test/slots-term-lengths";
    let dir = build_files_in(name, 20_000, ConcreteFileVersion::V2_2).await;
    let open = || {
        let dir = &dir;
        async move {
            let object_store = Arc::new(ObjectStore::local());
            let store = Arc::new(LanceIndexStore::new(object_store.clone(), dir.obj_path(), Arc::new(LanceCache::with_capacity(64 << 20))));
            let cache = LanceCache::with_capacity(64 << 20);
            let index = InvertedIndex::load(store, None, &cache).await.unwrap();
            object_store.io_stats_incremental();
            (object_store, index, cache)
        }
    };
    let (object_store, index, _cache) = open().await;
    let partition = &index.partitions()[0];
    partition.load_term_lengths().await.unwrap();
    let lengths_bytes = object_store.io_stats_incremental().read_bytes;
    let ids = all_token_ids(partition);
    let counts = partition.doc_freqs(&ids).await.unwrap();
    assert_eq!(object_store.io_stats_incremental().read_iops, 0, "counts are memory lookups once the lengths are loaded");

    let (object_store, index, _cache) = open().await;
    let partition = &index.partitions()[0];
    partition.load_term_metadata().await.unwrap();
    let metadata_bytes = object_store.io_stats_incremental().read_bytes;
    assert!(lengths_bytes < metadata_bytes, "lengths alone {lengths_bytes} bytes, with max scores {metadata_bytes}");
    assert_eq!(partition.doc_freqs(&ids).await.unwrap(), counts);
    assert!(unregister_tokenizer(name));
}
