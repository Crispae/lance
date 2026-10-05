// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prototype (RustIE RFC 0003 item 12): a per-partition side file holding each token's posting
//! blocks, positions and position offsets together in one packed-struct column, so that one read
//! of a token's row returns all three. The posting file keeps them in three columns, each read
//! with its own request. The side file is written offline from an existing partition and read only
//! by [`InvertedPartition::posting_cursors_from_sidecar`], to measure what co-located per-token data
//! saves in object-store reads. Lance's own search, merge and remap paths never see it.
//!
//! Row layout (byte strings, little endian): `posting` = block count (u32), each block's length
//! (u32), then the blocks; `positions` = the shared position stream as stored; `offsets` = the
//! position block offsets (u32 each).

use std::collections::HashMap;

use arrow_array::StructArray;
use arrow_schema::Fields;

use super::*;

/// The side file's one column.
const SIDECAR_COL: &str = "_rustie_term";
/// Rows copied per batch when writing the side file.
const SIDECAR_WRITE_ROWS: usize = 16_384;
/// About 3 bytes of posting and 2 of positions per document.
const SIDECAR_BYTES_PER_DOC: u64 = 5;

/// The side file of partition `partition_id`, next to its posting file.
pub fn sidecar_file_path(partition_id: u64) -> String {
    format!("part_{partition_id}_rustie_terms.lance")
}

fn sidecar_fields() -> Fields {
    Fields::from(vec![
        Field::new("posting", DataType::LargeBinary, false),
        Field::new("positions", DataType::LargeBinary, false),
        Field::new("offsets", DataType::LargeBinary, false),
    ])
}

fn sidecar_schema() -> Arc<Schema> {
    let packed = std::collections::HashMap::from([(lance_encoding::constants::PACKED_STRUCT_META_KEY.to_string(), "true".to_string())]);
    Arc::new(Schema::new(vec![Field::new(SIDECAR_COL, DataType::Struct(sidecar_fields()), false).with_metadata(packed)]))
}

fn encode_blocks(blocks: &LargeBinaryArray, out: &mut Vec<u8>) {
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks.iter() {
        out.extend_from_slice(&(block.map_or(0, <[u8]>::len) as u32).to_le_bytes());
    }
    for block in blocks.iter().flatten() {
        out.extend_from_slice(block);
    }
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
        .ok_or_else(|| Error::index("truncated RustIE side-file row".to_string()))
}

fn decode_blocks(bytes: &[u8]) -> Result<LargeBinaryArray> {
    let count = read_u32(bytes, 0)? as usize;
    let mut at = 4 + 4 * count;
    let mut blocks = Vec::with_capacity(count);
    for i in 0..count {
        let len = read_u32(bytes, 4 + 4 * i)? as usize;
        let block = bytes.get(at..at + len).ok_or_else(|| Error::index("truncated RustIE side-file row".to_string()))?;
        blocks.push(block);
        at += len;
    }
    Ok(LargeBinaryArray::from_iter_values(blocks))
}

fn decode_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect()
}

impl InvertedPartition {
    /// Writes this partition's side file (see the module docs) from its posting file. The index
    /// must have been built with positions, in the compressed layout.
    pub async fn write_term_sidecar(&self) -> Result<()> {
        let list = &self.inverted_list;
        list.ensure_bulk_layout()?;
        if !matches!(list.positions_layout, PositionsLayout::SharedStream(_)) {
            return Err(Error::invalid_input("the RustIE side file needs an index built with positions".to_string()));
        }
        let reader = list.reader.get().await?.clone();
        let num_rows = list.reader.num_rows();
        let schema = sidecar_schema();
        let mut writer = self.store.new_index_file(&sidecar_file_path(self.id), schema.clone()).await?;
        for start in (0..num_rows).step_by(SIDECAR_WRITE_ROWS) {
            let end = (start + SIDECAR_WRITE_ROWS).min(num_rows);
            let batch = reader.read_range(start..end, Some(&[POSTING_COL, COMPRESSED_POSITION_COL, POSITION_BLOCK_OFFSET_COL])).await?;
            let postings = batch[POSTING_COL].as_list::<i32>();
            let positions = batch[COMPRESSED_POSITION_COL].as_binary::<i64>();
            let offsets = batch[POSITION_BLOCK_OFFSET_COL].as_list::<i32>();
            let mut posting_rows = Vec::with_capacity(end - start);
            let mut offset_rows = Vec::with_capacity(end - start);
            for row in 0..batch.num_rows() {
                let mut bytes = Vec::new();
                encode_blocks(postings.value(row).as_binary::<i64>(), &mut bytes);
                posting_rows.push(bytes);
                let values = offsets.value(row);
                offset_rows.push(values.as_primitive::<UInt32Type>().values().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
            let column = StructArray::new(
                sidecar_fields(),
                vec![
                    Arc::new(LargeBinaryArray::from_iter_values(posting_rows)) as ArrayRef,
                    Arc::new(positions.clone()) as ArrayRef,
                    Arc::new(LargeBinaryArray::from_iter_values(offset_rows)) as ArrayRef,
                ],
                None,
            );
            writer.write_record_batch(RecordBatch::try_new(schema.clone(), vec![Arc::new(column)])?).await?;
        }
        writer.finish().await?;
        Ok(())
    }

    /// Cursors over `token_ids` (any order, repeats allowed), in that order, read from the side
    /// file: one request range serves a token's postings and positions together. Cached under the
    /// same keys as [`Self::posting_cursors`] (both always, so a later positional cursor is free);
    /// concurrent callers share reads.
    pub async fn posting_cursors_from_sidecar(&self, token_ids: &[u32], with_positions: bool) -> Result<Vec<PostingCursor>> {
        let list = &self.inverted_list;
        list.ensure_bulk_layout()?;
        let PositionsLayout::SharedStream(codec) = list.positions_layout else {
            return Err(Error::invalid_input("the RustIE side file needs an index built with positions".to_string()));
        };
        let reader = list
            .sidecar
            .get_or_try_init(|| async { self.store.open_index_file(&sidecar_file_path(self.id)).await })
            .await?
            .clone();
        let mut lists: HashMap<u32, CompressedPostingList> = HashMap::new();
        let mut positions: HashMap<u32, CompressedPositionStorage> = HashMap::new();
        let mut pending: Vec<u32> = token_ids.to_vec();
        pending.sort_unstable();
        pending.dedup();
        while !pending.is_empty() {
            let mut missing = Vec::new();
            for &token in &pending {
                let cached_list = list.index_cache.get_with_key(&TermListKey { token_id: token }).await;
                let cached_positions = if with_positions { list.index_cache.get_with_key(&PositionKey { token_id: token }).await } else { None };
                match (cached_list, cached_positions) {
                    (Some(hit), Some(pos)) => {
                        lists.insert(token, hit.0.clone());
                        positions.insert(token, pos.0.clone());
                    }
                    (Some(hit), None) if !with_positions => {
                        lists.insert(token, hit.0.clone());
                    }
                    _ => missing.push(token),
                }
            }
            if missing.is_empty() {
                break;
            }
            let (claim, waits) = list.bulk_in_flight.claim(false, &missing);
            if let Some(claim) = claim {
                list.read_sidecar_rows(&reader, &claim.tokens, codec, &mut lists, &mut positions).await?;
                drop(claim);
            }
            pending = missing.into_iter().filter(|token| !lists.contains_key(token) || (with_positions && !positions.contains_key(token))).collect();
            wait_for_loads(waits).await;
        }
        token_ids
            .iter()
            .map(|token| {
                let mut posting = lists[token].clone();
                if with_positions {
                    posting.positions = Some(positions[token].clone());
                }
                PostingCursor::new(posting, with_positions)
            })
            .collect()
    }
}

impl PostingListReader {
    /// Reads the side-file rows of `tokens` (nearby rows in one range), caching each token's posting
    /// list and positions.
    async fn read_sidecar_rows(
        &self,
        reader: &Arc<dyn IndexReader>,
        tokens: &[u32],
        codec: PositionStreamCodec,
        lists: &mut HashMap<u32, CompressedPostingList>,
        positions: &mut HashMap<u32, CompressedPositionStorage>,
    ) -> Result<()> {
        let mut tokens = tokens.to_vec();
        tokens.sort_unstable();
        let lengths = self.bulk_lengths(&tokens).await?;
        let resident = self.known_lengths();
        let ranges = coalesce_rows(&tokens, |row| match (lengths.get(&row), resident) {
            (Some(length), _) => 32 + SIDECAR_BYTES_PER_DOC * u64::from(*length),
            (None, Some(resident)) => 32 + SIDECAR_BYTES_PER_DOC * u64::from(resident[row as usize]),
            (None, None) => BULK_MAX_GAP_BYTES + 1,
        });
        let mut batches: Vec<(usize, RecordBatch)> = stream::iter(ranges.iter().cloned().enumerate().map(|(i, range)| {
            let reader = reader.clone();
            async move { Ok::<_, Error>((i, reader.read_range(range.start as usize..range.end as usize, None).await?)) }
        }))
        .buffer_unordered(BULK_READ_CONCURRENCY)
        .try_collect()
        .await?;
        batches.sort_unstable_by_key(|(i, _)| *i);
        for &token in &tokens {
            let at = ranges.partition_point(|range| range.end <= token);
            let row = (token - ranges[at].start) as usize;
            let column = batches[at].1[SIDECAR_COL].as_struct();
            let blocks = decode_blocks(column.column(0).as_binary::<i64>().value(row))?;
            let list = CompressedPostingList::new(blocks, 0.0, lengths[&token], self.posting_tail_codec, self.block_size, None, None);
            if !self.modern_posting_is_validated(token)? {
                self.ensure_modern_posting_validated(token, &PostingList::Compressed(list.clone())).await?;
            }
            let stream_bytes = bytes::Bytes::from(column.column(1).as_binary::<i64>().value(row).to_vec());
            let block_offsets = decode_u32s(column.column(2).as_binary::<i64>().value(row));
            let storage = CompressedPositionStorage::SharedStream(SharedPositionStream::new(codec, block_offsets, stream_bytes));
            self.index_cache.insert_with_key(&TermListKey { token_id: token }, Arc::new(TermList(list.clone()))).await;
            self.index_cache.insert_with_key(&PositionKey { token_id: token }, Arc::new(Positions(storage.clone()))).await;
            lists.insert(token, list);
            positions.insert(token, storage);
        }
        Ok(())
    }
}
