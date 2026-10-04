// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bulk reads of many tokens' postings and positions, for callers that open thousands of cursors
//! at once (a regex expanding to a large part of the dictionary).
//!
//! The per-token path costs several object-store requests per token: a document-frequency read of
//! two columns that sit megabytes apart, a read of the token's whole 128-token cache group (which
//! rarely helps when the tokens are scattered over the dictionary), and a separate read of its
//! positions. Here the requested rows are sorted and nearby ones are read as one range, so a query
//! costs a handful of requests whatever the number of tokens. The results are cached per token
//! (postings under [`TermListKey`], positions under the existing [`PositionKey`]), so repeated
//! queries are served from memory.

use std::collections::HashMap;
use std::ops::Range;

use super::*;

/// Rows closer than this many bytes (estimated) are read in one range, discarding what lies between
/// them: one more request costs more than reading this much extra.
const BULK_MAX_GAP_BYTES: u64 = 512 * 1024;
/// A range stops growing at the size Lance reads in one piece anyway.
const BULK_MAX_RANGE_BYTES: u64 = 8 * 1024 * 1024;
/// Ranges read concurrently.
const BULK_READ_CONCURRENCY: usize = 16;

/// Groups the ascending, distinct `rows` into ranges to read, each at most [`BULK_MAX_RANGE_BYTES`]
/// and never bridging more than [`BULK_MAX_GAP_BYTES`] of unwanted rows, by the byte estimate
/// `bytes_of`. Every requested row is inside exactly one range.
pub(super) fn coalesce_rows(rows: &[u32], bytes_of: impl Fn(u32) -> u64) -> Vec<Range<u32>> {
    let mut ranges: Vec<Range<u32>> = Vec::new();
    let mut range_bytes = 0u64;
    for &row in rows {
        let own = bytes_of(row);
        if let Some(last) = ranges.last_mut() {
            let mut gap = 0u64;
            let mut skipped = last.end;
            while skipped < row && gap <= BULK_MAX_GAP_BYTES {
                gap += bytes_of(skipped);
                skipped += 1;
            }
            if gap <= BULK_MAX_GAP_BYTES && range_bytes + gap + own <= BULK_MAX_RANGE_BYTES {
                last.end = row + 1;
                range_bytes += gap + own;
                continue;
            }
        }
        ranges.push(row..row + 1);
        range_bytes = own;
    }
    ranges
}

/// Bulk loads in progress, so callers wanting the same token at the same time share one read
/// (the cache alone does not do that: all of them miss, and each reads the row again).
#[derive(Default)]
pub(super) struct BulkInFlight {
    /// `(is_positions, token)` -> a receiver that turns `true` once the owning call is done.
    loading: std::sync::Mutex<HashMap<(bool, u32), tokio::sync::watch::Receiver<bool>>>,
}

/// The tokens one call has claimed; released (and waiters woken) when dropped, including when the
/// call fails or is cancelled.
struct Claim<'a> {
    registry: &'a BulkInFlight,
    is_positions: bool,
    tokens: Vec<u32>,
    done: tokio::sync::watch::Sender<bool>,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        let mut loading = self.registry.loading.lock().unwrap();
        for token in &self.tokens {
            loading.remove(&(self.is_positions, *token));
        }
        drop(loading);
        self.done.send_replace(true);
    }
}

impl BulkInFlight {
    /// Claims every token of `tokens` that nobody is loading; for the others returns what to wait on.
    fn claim<'a>(&'a self, is_positions: bool, tokens: &[u32]) -> (Option<Claim<'a>>, Vec<tokio::sync::watch::Receiver<bool>>) {
        let (done, receiver) = tokio::sync::watch::channel(false);
        let mut loading = self.loading.lock().unwrap();
        let mut mine = Vec::new();
        let mut waits = Vec::new();
        for &token in tokens {
            match loading.get(&(is_positions, token)) {
                Some(other) => waits.push(other.clone()),
                None => {
                    loading.insert((is_positions, token), receiver.clone());
                    mine.push(token);
                }
            }
        }
        drop(loading);
        let claim = (!mine.is_empty()).then_some(Claim { registry: self, is_positions, tokens: mine, done });
        (claim, waits)
    }
}

async fn wait_for_loads(waits: Vec<tokio::sync::watch::Receiver<bool>>) {
    for mut receiver in waits {
        // An error means the owner is gone, which is also a release.
        let _ = receiver.wait_for(|done| *done).await;
    }
}

/// A token's posting list as read by the bulk path: doc ids and frequencies only (no impacts, no
/// positions; those are separate).
#[derive(Debug, Clone, DeepSizeOf)]
pub struct TermList(pub(super) CompressedPostingList);

#[derive(Debug, Clone)]
pub struct TermListKey {
    pub token_id: u32,
}

impl CacheKey for TermListKey {
    type ValueType = TermList;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("term-list-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "TermList"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.term-list-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.token_id);
    }
}

/// A token's number of documents as read by the bulk path (just the `_length` column).
#[derive(Debug, Clone, DeepSizeOf)]
pub struct TermLength(pub(super) u32);

#[derive(Debug, Clone)]
pub struct TermLengthKey {
    pub token_id: u32,
}

impl CacheKey for TermLengthKey {
    type ValueType = TermLength;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("term-length-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "TermLength"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.term-length-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.token_id);
    }
}

impl PostingListReader {
    /// The bulk path needs the compressed (v2) posting layout.
    fn ensure_bulk_layout(&self) -> Result<()> {
        if self.is_legacy_layout() || !matches!(self.metadata, PostingMetadata::V2 { .. }) {
            return Err(Error::index("bulk posting reads need the v2 posting layout; rebuild the legacy index".to_string()));
        }
        Ok(())
    }

    /// The per-token `(max_score, length)` columns when they are resident (see
    /// [`InvertedPartition::load_term_metadata`]); the bulk path works without them.
    fn resident_metadata(&self) -> Option<&LoadedPostingMetadata> {
        match &self.metadata {
            PostingMetadata::V2 { metadata } => metadata.get(),
            PostingMetadata::LegacyV1 { .. } => None,
        }
    }

    /// The number of documents of each of `token_ids` (distinct or not). Resident metadata answers
    /// from memory; otherwise only the `_length` rows needed are read (4 bytes per row, nearby rows
    /// in one request) and cached per token, instead of two requests per token.
    pub(super) async fn bulk_lengths(&self, token_ids: &[u32]) -> Result<HashMap<u32, u32>> {
        self.ensure_bulk_layout()?;
        let resident = self.resident_metadata();
        let mut out: HashMap<u32, u32> = HashMap::new();
        let mut missing: Vec<u32> = Vec::new();
        let mut distinct = token_ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        for token in distinct {
            if let Some(resident) = resident {
                out.insert(token, resident.lengths[token as usize]);
            } else if let Some(hit) = self.index_cache.get_with_key(&TermLengthKey { token_id: token }).await {
                out.insert(token, hit.0);
            } else {
                missing.push(token);
            }
        }
        if !missing.is_empty() {
            let ranges = coalesce_rows(&missing, |_| 4);
            let batches = self.read_row_ranges(&ranges, &[LENGTH_COL]).await?;
            for &token in &missing {
                let at = ranges.partition_point(|range| range.end <= token);
                let row = (token - ranges[at].start) as usize;
                let length = batches[at][LENGTH_COL].as_primitive::<UInt32Type>().value(row);
                self.index_cache.insert_with_key(&TermLengthKey { token_id: token }, Arc::new(TermLength(length))).await;
                out.insert(token, length);
            }
        }
        Ok(out)
    }

    /// Reads `columns` of every range, in order, with bounded concurrency.
    async fn read_row_ranges(&self, ranges: &[Range<u32>], columns: &[&str]) -> Result<Vec<RecordBatch>> {
        let reader = self.reader.get().await?.clone();
        let mut batches: Vec<(usize, RecordBatch)> = stream::iter(ranges.iter().cloned().enumerate().map(|(i, range)| {
            let reader = reader.clone();
            async move {
                let batch = reader.read_range(range.start as usize..range.end as usize, Some(columns)).await?;
                Ok::<_, Error>((i, batch))
            }
        }))
        .buffer_unordered(BULK_READ_CONCURRENCY)
        .try_collect()
        .await?;
        batches.sort_unstable_by_key(|(i, _)| *i);
        Ok(batches.into_iter().map(|(_, batch)| batch).collect())
    }

    /// The posting lists of `token_ids` (any order, repeats allowed), in that order. Lists not in
    /// the cache are read with their neighbours in few requests; concurrent callers wanting the same
    /// token share one read.
    pub(super) async fn bulk_term_lists(&self, token_ids: &[u32], metrics: &dyn MetricsCollector) -> Result<Vec<CompressedPostingList>> {
        self.ensure_bulk_layout()?;
        let mut found: HashMap<u32, CompressedPostingList> = HashMap::new();
        let mut pending: Vec<u32> = token_ids.to_vec();
        pending.sort_unstable();
        pending.dedup();
        while !pending.is_empty() {
            let mut missing: Vec<u32> = Vec::new();
            for &token in &pending {
                match self.index_cache.get_with_key(&TermListKey { token_id: token }).await {
                    Some(hit) => {
                        metrics.record_index_cache_hit();
                        found.insert(token, hit.0.clone());
                    }
                    None => {
                        metrics.record_index_cache_miss();
                        missing.push(token);
                    }
                }
            }
            if missing.is_empty() {
                break;
            }
            let (claim, waits) = self.bulk_in_flight.claim(false, &missing);
            if let Some(claim) = claim {
                self.read_term_lists(&claim.tokens, &mut found).await?;
                drop(claim);
            }
            // What another call was loading is in the cache by now (or is claimed next round).
            pending = missing.into_iter().filter(|token| !found.contains_key(token)).collect();
            wait_for_loads(waits).await;
        }
        Ok(token_ids.iter().map(|token| found[token].clone()).collect())
    }

    /// Reads the posting rows of `tokens`, caching and returning each.
    async fn read_term_lists(&self, tokens: &[u32], found: &mut HashMap<u32, CompressedPostingList>) -> Result<()> {
        let mut tokens = tokens.to_vec();
        tokens.sort_unstable();
        let lengths = self.bulk_lengths(&tokens).await?;
        let resident = self.resident_metadata();
        // A posting row is roughly 3 bytes per document (block-packed ids and frequencies). Rows
        // whose length is not known (only between wanted ones, without resident metadata) are never
        // bridged, so nothing unknown is read and discarded.
        let ranges = coalesce_rows(&tokens, |row| match (lengths.get(&row), resident) {
            (Some(length), _) => 32 + 3 * u64::from(*length),
            (None, Some(resident)) => 32 + 3 * u64::from(resident.lengths[row as usize]),
            (None, None) => BULK_MAX_GAP_BYTES + 1,
        });
        let batches = self.read_row_ranges(&ranges, &[POSTING_COL]).await?;
        for &token in &tokens {
            let at = ranges.partition_point(|range| range.end <= token);
            let row = (token - ranges[at].start) as usize;
            // A copy of just this token's row: the cache must not pin the whole range's buffers.
            let one = batches[at].slice(row, 1).shrink_to_fit()?;
            // `max_score` only matters to BM25 ranking, which cursors do not use.
            let max_score = resident.map_or(0.0, |resident| resident.max_scores[token as usize]);
            let list = CompressedPostingList::from_batch(&one, max_score, lengths[&token], self.posting_tail_codec, self.block_size, None)?;
            if !self.modern_posting_is_validated(token)? {
                self.ensure_modern_posting_validated(token, &PostingList::Compressed(list.clone())).await?;
            }
            self.index_cache.insert_with_key(&TermListKey { token_id: token }, Arc::new(TermList(list.clone()))).await;
            found.insert(token, list);
        }
        Ok(())
    }

    /// The positions of `token_ids` (any order, repeats allowed), in that order, read like
    /// [`Self::bulk_term_lists`] (including sharing reads between concurrent callers) and cached under
    /// the same keys the per-token path uses.
    pub(super) async fn bulk_positions(&self, token_ids: &[u32], metrics: &dyn MetricsCollector) -> Result<Vec<CompressedPositionStorage>> {
        let PositionsLayout::SharedStream(codec) = self.positions_layout else {
            return Err(Error::invalid_input(
                "bulk position reads need the shared position stream; the index was built without positions or in a legacy layout".to_string(),
            ));
        };
        self.ensure_bulk_layout()?;
        let mut found: HashMap<u32, CompressedPositionStorage> = HashMap::new();
        let mut pending: Vec<u32> = token_ids.to_vec();
        pending.sort_unstable();
        pending.dedup();
        while !pending.is_empty() {
            let mut missing: Vec<u32> = Vec::new();
            for &token in &pending {
                match self.index_cache.get_with_key(&PositionKey { token_id: token }).await {
                    Some(hit) => {
                        metrics.record_index_cache_hit();
                        found.insert(token, hit.0.clone());
                    }
                    None => {
                        metrics.record_index_cache_miss();
                        missing.push(token);
                    }
                }
            }
            if missing.is_empty() {
                break;
            }
            let (claim, waits) = self.bulk_in_flight.claim(true, &missing);
            if let Some(claim) = claim {
                self.read_positions_rows(&claim.tokens, codec, &mut found).await?;
                drop(claim);
            }
            pending = missing.into_iter().filter(|token| !found.contains_key(token)).collect();
            wait_for_loads(waits).await;
        }
        Ok(token_ids.iter().map(|token| found[token].clone()).collect())
    }

    /// Reads the position rows of `tokens`, caching and returning each.
    async fn read_positions_rows(
        &self,
        tokens: &[u32],
        codec: PositionStreamCodec,
        found: &mut HashMap<u32, CompressedPositionStorage>,
    ) -> Result<()> {
        let mut tokens = tokens.to_vec();
        tokens.sort_unstable();
        let lengths = self.bulk_lengths(&tokens).await?;
        let resident = self.resident_metadata();
        // Positions take about a byte and a half each, and a posting holds a bit more than one.
        let ranges = coalesce_rows(&tokens, |row| match (lengths.get(&row), resident) {
            (Some(length), _) => 32 + 2 * u64::from(*length),
            (None, Some(resident)) => 32 + 2 * u64::from(resident.lengths[row as usize]),
            (None, None) => BULK_MAX_GAP_BYTES + 1,
        });
        let batches = self.read_row_ranges(&ranges, &[COMPRESSED_POSITION_COL, POSITION_BLOCK_OFFSET_COL]).await?;
        for &token in &tokens {
            let at = ranges.partition_point(|range| range.end <= token);
            let row = (token - ranges[at].start) as usize;
            let batch = &batches[at];
            let bytes = bytes::Bytes::from(batch[COMPRESSED_POSITION_COL].as_binary::<i64>().value(row).to_vec());
            let block_offsets = batch[POSITION_BLOCK_OFFSET_COL].as_list::<i32>().value(row).as_primitive::<UInt32Type>().values().to_vec();
            let storage = CompressedPositionStorage::SharedStream(SharedPositionStream::new(codec, block_offsets, bytes));
            self.index_cache.insert_with_key(&PositionKey { token_id: token }, Arc::new(Positions(storage.clone()))).await;
            found.insert(token, storage);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalescing_merges_close_rows_and_splits_far_or_large_ones() {
        let small = |_: u32| 100u64;
        // Adjacent and nearby rows are one range; the unwanted rows between them are read too.
        assert_eq!(coalesce_rows(&[3, 4, 5, 9], small), vec![3..10]);
        assert_eq!(coalesce_rows(&[], small), Vec::<Range<u32>>::new());
        assert_eq!(coalesce_rows(&[7], small), vec![7..8]);
        // A gap of more than BULK_MAX_GAP_BYTES (here 6,000 rows of 100 bytes) starts a new range.
        assert_eq!(coalesce_rows(&[0, 1, 7_000, 7_001], small), vec![0..2, 7_000..7_002]);
        // Rows are never dropped or reordered: every requested row lies in exactly one range.
        let rows: Vec<u32> = (0..2_000).map(|i| i * 37 % 100_000).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let ranges = coalesce_rows(&rows, small);
        for row in &rows {
            assert_eq!(ranges.iter().filter(|r| r.contains(row)).count(), 1, "row {row}");
        }
        assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
        // A range stops growing at BULK_MAX_RANGE_BYTES: big rows each get their own range.
        let big = |_: u32| BULK_MAX_RANGE_BYTES;
        assert_eq!(coalesce_rows(&[1, 2, 3], big), vec![1..2, 2..3, 3..4]);
        let half = |_: u32| BULK_MAX_RANGE_BYTES / 2;
        assert_eq!(coalesce_rows(&[1, 2, 3], half), vec![1..3, 3..4]);
    }
}
