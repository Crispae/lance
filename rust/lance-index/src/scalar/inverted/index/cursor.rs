// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A seekable cursor over one token's compressed posting list: document ids, frequencies and
//! positions, for callers that run their own query logic (conjunctions, same-position checks)
//! instead of the BM25 scorers.

use super::*;

/// The doc id a cursor reports once it has run past the last document.
pub const TERMINATED: u32 = u32::MAX;

/// Walks the documents of one token's posting list in ascending doc-id order. Doc ids are local
/// to the partition the list came from.
///
/// Only the blocks a walk touches are decoded: a [`PostingCursor::seek`] reads the block head
/// table, then decodes the doc ids of the single block that can hold the target. Frequencies and
/// positions are decoded lazily, a block at a time, when asked for.
pub struct PostingCursor {
    list: CompressedPostingList,
    num_blocks: usize,
    /// Documents in the last block when it is a partial one, else 0.
    remainder: usize,
    with_positions: bool,
    /// The block the cursor is in; `num_blocks` once exhausted.
    block: usize,
    /// Doc ids of the current block.
    doc_ids: Vec<u32>,
    /// Byte offset of the current block's frequency stream.
    freq_offset: usize,
    freqs: Vec<u32>,
    freqs_ready: bool,
    /// Index of the current document within the block.
    pos: usize,
    /// The block `positions` / `position_offsets` were decoded for.
    positions_block: Option<usize>,
    positions: Vec<u32>,
    position_offsets: Vec<usize>,
    scratch: Vec<u32>,
}

impl PostingCursor {
    /// A cursor positioned on the first document. `with_positions` requires the list to carry a
    /// shared position stream (the V2/V3 layout); legacy layouts are rejected.
    pub fn new(list: CompressedPostingList, with_positions: bool) -> Result<Self> {
        if with_positions && !matches!(list.positions, Some(CompressedPositionStorage::SharedStream(_))) {
            return Err(Error::index(
                "the posting list has no shared position stream (legacy layout, or loaded without positions)"
                    .to_string(),
            ));
        }
        let num_blocks = list.blocks.len();
        let remainder = list.length as usize % list.block_size;
        let scratch = vec![0u32; list.block_size];
        let mut cursor = Self {
            list,
            num_blocks,
            remainder,
            with_positions,
            block: 0,
            doc_ids: Vec::new(),
            freq_offset: 0,
            freqs: Vec::new(),
            freqs_ready: false,
            pos: 0,
            positions_block: None,
            positions: Vec::new(),
            position_offsets: Vec::new(),
            scratch,
        };
        cursor.load_block(0);
        Ok(cursor)
    }

    /// Number of documents in the list.
    pub fn len(&self) -> usize {
        self.list.length as usize
    }

    pub fn is_empty(&self) -> bool {
        self.list.length == 0
    }

    /// The current doc id, or [`TERMINATED`].
    #[inline]
    pub fn doc(&self) -> u32 {
        if self.block >= self.num_blocks {
            TERMINATED
        } else {
            self.doc_ids[self.pos]
        }
    }

    /// Moves to the next document and returns it, or [`TERMINATED`].
    pub fn advance(&mut self) -> u32 {
        if self.block >= self.num_blocks {
            return TERMINATED;
        }
        self.pos += 1;
        if self.pos >= self.doc_ids.len() {
            self.load_block(self.block + 1);
        }
        self.doc()
    }

    /// Moves to the first document `>= target` and returns it, or [`TERMINATED`]. A target at or
    /// before the current document leaves the cursor where it is: a cursor never moves backwards.
    pub fn seek(&mut self, target: u32) -> u32 {
        if self.block >= self.num_blocks {
            return TERMINATED;
        }
        if self.doc() >= target {
            return self.doc();
        }
        // The only block that can hold `target` is the last one starting at or before it.
        let first_docs = self.list.block_first_docs();
        let block = first_docs
            .partition_point(|&first| first <= target)
            .saturating_sub(1)
            .max(self.block);
        if block != self.block {
            self.load_block(block);
        }
        let at = self.pos + self.doc_ids[self.pos..].partition_point(|&doc| doc < target);
        if at < self.doc_ids.len() {
            self.pos = at;
        } else {
            // Every document of this block is below `target`; the next block starts above it.
            self.load_block(block + 1);
        }
        self.doc()
    }

    /// The current document's frequency (occurrences of the token in it).
    pub fn freq(&mut self) -> u32 {
        debug_assert!(self.block < self.num_blocks, "freq() on an exhausted cursor");
        self.ensure_freqs();
        self.freqs[self.pos]
    }

    /// The current document's token positions, ascending, into `out` (cleared first).
    pub fn positions(&mut self, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        if !self.with_positions {
            return Err(Error::index(
                "positions() on a cursor opened without positions".to_string(),
            ));
        }
        if self.block >= self.num_blocks {
            return Ok(());
        }
        self.ensure_freqs();
        if self.positions_block != Some(self.block) {
            let Some(CompressedPositionStorage::SharedStream(stream)) = &self.list.positions else {
                unreachable!("checked in new()")
            };
            self.positions.clear();
            super::super::encoding::decode_position_stream_block(
                stream.block(self.block),
                &self.freqs,
                stream.codec(),
                &mut self.positions,
            )?;
            self.position_offsets.clear();
            self.position_offsets.push(0);
            let mut total = 0usize;
            for &freq in &self.freqs {
                total += freq as usize;
                self.position_offsets.push(total);
            }
            if total != self.positions.len() {
                return Err(Error::index(format!(
                    "block {} has {} positions but its frequencies sum to {total}",
                    self.block,
                    self.positions.len()
                )));
            }
            self.positions_block = Some(self.block);
        }
        out.extend_from_slice(&self.positions[self.position_offsets[self.pos]..self.position_offsets[self.pos + 1]]);
        Ok(())
    }

    fn is_tail_block(&self, block: usize) -> bool {
        block + 1 == self.num_blocks && self.remainder != 0
    }

    /// Positions the cursor on the first document of `block`; past the last block it becomes
    /// exhausted.
    fn load_block(&mut self, block: usize) {
        self.block = block;
        self.pos = 0;
        self.doc_ids.clear();
        self.freqs.clear();
        self.freqs_ready = false;
        if block >= self.num_blocks {
            self.block = self.num_blocks;
            return;
        }
        let bytes = self.list.blocks.value(block);
        self.freq_offset = if self.is_tail_block(block) {
            super::super::encoding::decompress_posting_remainder_doc_ids(
                bytes,
                self.remainder,
                self.list.posting_tail_codec,
                self.list.block_size,
                &mut self.doc_ids,
            )
        } else {
            super::super::encoding::decompress_posting_block_doc_ids(
                bytes,
                &mut self.scratch,
                &mut self.doc_ids,
                self.list.block_size,
            )
        };
    }

    fn ensure_freqs(&mut self) {
        if self.freqs_ready {
            return;
        }
        let bytes = self.list.blocks.value(self.block);
        self.freqs.clear();
        if self.is_tail_block(self.block) {
            super::super::encoding::decompress_posting_remainder_frequencies(
                bytes,
                self.freq_offset,
                self.remainder,
                self.list.posting_tail_codec,
                &mut self.freqs,
            );
        } else {
            super::super::encoding::decompress_posting_block_frequencies(
                bytes,
                self.freq_offset,
                &mut self.scratch,
                &mut self.freqs,
                self.list.block_size,
            );
        }
        self.freqs_ready = true;
    }
}
