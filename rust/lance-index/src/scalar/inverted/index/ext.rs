// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read access to an FTS index's internals for callers that run their own query logic: the
//! partitions of an index, each partition's token dictionary, document frequencies, posting
//! cursors, and the row ids of its documents. Nothing here changes how the index is built or
//! searched.

use super::*;

impl InvertedIndex {
    /// The index's partitions. Each has its own token dictionary and its own doc-id space, so a
    /// query is evaluated per partition and the matching rows are combined afterwards.
    pub fn partitions(&self) -> &[Arc<InvertedPartition>] {
        &self.partitions
    }
}

/// The row ids of one partition's documents, resident in memory.
pub struct PartitionRows(AddressKeyedDocuments);

impl PartitionRows {
    /// Number of documents, counting dead slots a remapped partition keeps so that doc ids stay
    /// aligned with its posting lists.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The row id of `doc_id` as the index stores it (a stable row id when the dataset has them,
    /// else a row address), or `None` for a dead slot.
    #[inline]
    pub fn row_id(&self, doc_id: u32) -> Option<u64> {
        let row = self.0.row_address(doc_id);
        (row != RowAddress::TOMBSTONE_ROW).then_some(row)
    }
}

impl InvertedPartition {
    /// The token dictionary as an FST, for automaton searches (regex, prefix). `None` only for a
    /// partition that is still being built; loaded partitions always have one.
    pub fn token_fst(&self) -> Option<&fst::Map<Vec<u8>>> {
        match &self.tokens.tokens {
            TokenMap::Fst(map) => Some(map),
            TokenMap::HashMap(_) => None,
        }
    }

    /// The id of an exact token, or `None` when the partition never saw it.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokens.get(token)
    }

    /// Number of documents holding the token, without reading its posting list. Counts documents
    /// of deleted rows too.
    pub async fn doc_freq(&self, token_id: u32, metrics: Option<&dyn MetricsCollector>) -> Result<u32> {
        let length = self.inverted_list.posting_len_for_token(token_id, metrics).await?;
        u32::try_from(length).map_err(|_| Error::index(format!("posting length {length} exceeds u32")))
    }

    /// A cursor over the token's documents. `with_positions` also loads the token's positions.
    /// Partitions in the legacy layout are rejected.
    pub async fn posting_cursor(
        &self,
        token_id: u32,
        with_positions: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<PostingCursor> {
        match self.inverted_list.posting_list(token_id, with_positions, metrics).await? {
            PostingList::Compressed(list) => PostingCursor::new(list, with_positions),
            PostingList::Plain(_) => Err(Error::index(
                "posting cursors need the compressed posting layout; rebuild the legacy index".to_string(),
            )),
        }
    }

    /// Loads the per-token document counts (and scores) of the whole partition in two requests, so
    /// [`Self::doc_freq`] is answered from memory afterwards. Worth it once a query looks up more than
    /// a few dozen tokens (a regex expansion): it costs about 8 bytes per dictionary token, once,
    /// instead of two requests per token. Not for point queries.
    pub async fn load_term_metadata(&self) -> Result<()> {
        self.inverted_list.ensure_metadata_loaded().await
    }

    /// Cursors over many tokens' documents, in the order of `token_ids` (repeats allowed), read in a
    /// few coalesced requests instead of several per token, and cached per token.
    ///
    /// Use it when a query opens many cursors at once (thousands of terms of a regex): requests grow
    /// with the spread of the tokens in the dictionary, not with their number. It loads the
    /// partition's per-token metadata (see [`Self::load_term_metadata`]); for a handful of tokens
    /// [`Self::posting_cursor`] is cheaper. Rows between wanted ones may be read and discarded, so it
    /// reads somewhat more bytes than the per-token path. Partitions in the legacy layout are rejected.
    pub async fn posting_cursors(
        &self,
        token_ids: &[u32],
        with_positions: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<PostingCursor>> {
        let mut lists = self.inverted_list.bulk_term_lists(token_ids, metrics).await?;
        if with_positions {
            let positions = self.inverted_list.bulk_positions(token_ids, metrics).await?;
            for (list, positions) in lists.iter_mut().zip(positions) {
                list.positions = Some(positions);
            }
        }
        lists.into_iter().map(|list| PostingCursor::new(list, with_positions)).collect()
    }

    /// The partition's doc id → row id table. The first call loads the whole table; keep the
    /// result for the lifetime of the query (or longer).
    pub async fn rows(&self) -> Result<PartitionRows> {
        Ok(PartitionRows(self.docs.address_keyed().await?))
    }
}
