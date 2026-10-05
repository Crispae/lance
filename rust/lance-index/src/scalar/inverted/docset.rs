// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scoring-free query building blocks over posting lists, for callers that need *every* match of
//! a boolean/positional query (no ranking, no top-k).
//!
//! The pieces are the document-at-a-time iterators BM25 search already composes, with the scoring
//! half of their protocol left out:
//!
//! - [`DocIterator`]: a sorted, seekable stream of partition-local document ids with a two-phase
//!   `matches` check. [`PostingCursor`] implements it.
//! - [`conjunction`] / [`disjunction`]: n-ary AND / OR over boxed iterators. The conjunction drives
//!   from the cheapest child (`cost`) and runs the expensive `matches` checks cheapest-first
//!   (`match_cost`), only on documents every child agrees on.
//! - [`align`]: the leapfrog step of a conjunction over concrete children, for iterators that need
//!   to keep access to their children (e.g. to read their positions in `matches`).
//! - [`positions_align`]: whether term position lists line up at fixed offsets (a phrase when the
//!   offsets are `0, 1, 2, ..`, "the same token" when they are all `0`).

pub use super::compound::DocIterator;
use super::compound::{
    DisjunctionScore, DisjunctionScorer, RequiredConjunctionScorer, align_conjunction_children,
};
use super::index::PostingCursor;
use lance_core::Result;

/// The documents every child yields. Children are advanced cheapest-first and `matches` is asked
/// of each only for documents all children contain. At least one child is required.
pub fn conjunction<'a>(
    children: Vec<Box<dyn DocIterator + 'a>>,
) -> Result<Box<dyn DocIterator + 'a>> {
    Ok(Box::new(RequiredConjunctionScorer::try_new(children)?))
}

/// The documents any child yields; `matches` holds when some child positioned on the document
/// matches. At least one child is required.
pub fn disjunction<'a>(
    children: Vec<Box<dyn DocIterator + 'a>>,
) -> Result<Box<dyn DocIterator + 'a>> {
    Ok(Box::new(DisjunctionScorer::try_new(
        children,
        DisjunctionScore::Max,
    )?))
}

/// Moves every child to the first document `>= target` that all of them contain and returns it
/// (`None` once any child is exhausted). Children are advanced in the order given, so list the
/// cheapest first.
pub fn align<C: DocIterator + ?Sized>(children: &mut [Box<C>], target: u64) -> Result<Option<u64>> {
    align_conjunction_children(children, target, |position| position)
}

/// Whether there is a base `b` such that, for every `(positions, offset)` clause, `b + offset` is in
/// `positions`. Each list must be ascending. Offsets `[0, 1, 2, ..]` check a phrase, all-zero
/// offsets check that one token carries every term. The shortest list drives; the others are
/// probed by binary search, so the cost is about `min(len) * clauses * log(len)`.
pub fn positions_align(clauses: &[(&[u32], u32)]) -> bool {
    let Some(driver) = (0..clauses.len()).min_by_key(|&i| clauses[i].0.len()) else {
        return false;
    };
    let (driver_positions, driver_offset) = clauses[driver];
    driver_positions.iter().any(|&position| {
        let Some(base) = position.checked_sub(driver_offset) else {
            return false;
        };
        clauses.iter().enumerate().all(|(i, &(positions, offset))| {
            i == driver
                || base
                    .checked_add(offset)
                    .is_some_and(|want| positions.binary_search(&want).is_ok())
        })
    })
}

impl DocIterator for PostingCursor {
    fn doc(&self) -> Option<u64> {
        let doc = PostingCursor::doc(self);
        (doc != super::index::TERMINATED).then_some(u64::from(doc))
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let doc = PostingCursor::advance(self);
        Ok((doc != super::index::TERMINATED).then_some(u64::from(doc)))
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        let doc = match u32::try_from(target) {
            Ok(target) => PostingCursor::seek(self, target),
            Err(_) => return Ok(None),
        };
        Ok((doc != super::index::TERMINATED).then_some(u64::from(doc)))
    }

    fn cost(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model iterator over a plain sorted vector. Like the real ones it is unpositioned
    /// (`doc() == None`) until the first `next` / `advance`, and stays exhausted once done.
    struct Model {
        docs: Vec<u64>,
        /// `None` before the first move; `docs.len()` once exhausted.
        at: Option<usize>,
        match_cost: Option<f32>,
    }

    impl Model {
        fn new(docs: Vec<u64>) -> Self {
            Self {
                docs,
                at: None,
                match_cost: None,
            }
        }
        fn boxed(docs: Vec<u64>) -> Box<dyn DocIterator> {
            Box::new(Self::new(docs))
        }
    }

    impl DocIterator for Model {
        fn doc(&self) -> Option<u64> {
            self.at.and_then(|at| self.docs.get(at).copied())
        }
        fn next(&mut self) -> Result<Option<u64>> {
            self.at = Some(self.at.map_or(0, |at| (at + 1).min(self.docs.len())));
            Ok(self.doc())
        }
        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            let mut at = self.at.unwrap_or(0);
            while self.docs.get(at).is_some_and(|&doc| doc < target) {
                at += 1;
            }
            self.at = Some(at);
            Ok(self.doc())
        }
        fn cost(&self) -> usize {
            self.docs.len()
        }
        fn match_cost(&self) -> Option<f32> {
            self.match_cost
        }
    }

    fn drain(mut it: Box<dyn DocIterator>) -> Vec<u64> {
        let mut out = Vec::new();
        let mut doc = it.next().unwrap();
        while let Some(d) = doc {
            if it.matches().unwrap() {
                out.push(d);
            }
            doc = it.next().unwrap();
        }
        out
    }

    /// A small deterministic generator, so the tests need no extra dependency.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn random_docs(seed: &mut u64, density: u64) -> Vec<u64> {
        (0..400u64).filter(|_| lcg(seed) % 100 < density).collect()
    }

    #[test]
    fn conjunction_and_disjunction_match_a_set_model() {
        let mut seed = 7;
        for round in 0..60 {
            let lists: Vec<Vec<u64>> = (0..2 + round % 3)
                .map(|i| random_docs(&mut seed, 5 + 25 * (i as u64 % 3)))
                .collect();
            let and: Vec<u64> = lists[0]
                .iter()
                .copied()
                .filter(|d| lists[1..].iter().all(|l| l.contains(d)))
                .collect();
            let mut or: Vec<u64> = lists.iter().flatten().copied().collect();
            or.sort_unstable();
            or.dedup();
            let boxed = || lists.iter().cloned().map(Model::boxed).collect::<Vec<_>>();
            assert_eq!(
                drain(conjunction(boxed()).unwrap()),
                and,
                "AND round {round}"
            );
            assert_eq!(drain(disjunction(boxed()).unwrap()), or, "OR round {round}");
        }
    }

    #[test]
    fn conjunction_seeks_over_a_dense_child() {
        let dense = Model::boxed((0..1000).collect());
        let rare = Model::boxed(vec![3, 500, 999]);
        assert_eq!(
            drain(conjunction(vec![dense, rare]).unwrap()),
            [3, 500, 999]
        );
        assert!(conjunction(Vec::new()).is_err());
    }

    #[test]
    fn align_returns_the_first_common_document() {
        let mut children = vec![
            Box::new(Model::new(vec![2, 8, 20])),
            Box::new(Model::new(vec![1, 8, 9, 20])),
        ];
        assert_eq!(align(&mut children, 0).unwrap(), Some(8));
        assert_eq!(align(&mut children, 9).unwrap(), Some(20));
        assert_eq!(align(&mut children, 21).unwrap(), None);
    }

    #[test]
    fn positions_align_matches_a_brute_force_model() {
        let mut seed = 11;
        for _ in 0..500 {
            let lists: Vec<Vec<u32>> = (0..2 + lcg(&mut seed) % 3)
                .map(|_| {
                    let mut v: Vec<u32> = (0..20u32).filter(|_| lcg(&mut seed) % 4 == 0).collect();
                    v.dedup();
                    v
                })
                .collect();
            let offsets: Vec<u32> = (0..lists.len() as u32)
                .map(|i| if lcg(&mut seed) % 2 == 0 { 0 } else { i })
                .collect();
            let clauses: Vec<(&[u32], u32)> = lists
                .iter()
                .map(Vec::as_slice)
                .zip(offsets.iter().copied())
                .collect();
            let expected = (0..40u32).any(|base| {
                clauses
                    .iter()
                    .all(|(positions, offset)| positions.contains(&(base + offset)))
            });
            assert_eq!(positions_align(&clauses), expected, "{clauses:?}");
        }
        assert!(!positions_align(&[]));
    }
}
