use arrayvec::ArrayVec;
use blake2s_simd::{many::HashManyJob, Hash, Params};
use rand::seq::SliceRandom;
use rand::RngCore;

#[cfg(not(feature = "single"))]
use rayon::join;

#[cfg(feature = "single")]
fn join<T1, F1, T2, F2>(f1: F1, f2: F2) -> (T1, T2)
where
    F1: FnOnce() -> T1,
    F2: FnOnce() -> T2,
{
    (f1(), f2())
}

const HASH_SIZE: usize = 32;
const CHUNK_SIZE: usize = 4096;
const LARGE_CHUNK_SIZE: usize = 65536;
pub const BENCH_LENGTH: usize = 1 << 24; // about 17 MB

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
pub struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: Vec<usize>,
    offset_index: usize,
}

impl RandomInput {
    pub fn new(len: usize) -> Self {
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        rand::thread_rng().fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rand::thread_rng());
        Self {
            buf,
            len,
            offsets,
            offset_index: 0,
        }
    }

    pub fn get(&mut self) -> &[u8] {
        let offset = self.offsets[self.offset_index];
        self.offset_index += 1;
        if self.offset_index >= self.offsets.len() {
            self.offset_index = 0;
        }
        &self.buf[offset..][..self.len]
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2b_load;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2s_load;

fn largest_power_of_two_leq(n: usize) -> usize {
    ((n / 2) + 1).next_power_of_two()
}

fn left_len(content_len: usize) -> usize {
    debug_assert!(content_len > CHUNK_SIZE);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / CHUNK_SIZE;
    largest_power_of_two_leq(full_chunks) * CHUNK_SIZE
}

#[derive(Clone, Copy, Debug)]
pub enum Finalization {
    NotRoot,
    Root,
}
use self::Finalization::{NotRoot, Root};

fn common_params() -> Params {
    let mut params = Params::new();
    params
        .hash_length(HASH_SIZE)
        .fanout(2)
        .max_depth(64)
        .max_leaf_length(CHUNK_SIZE as u32)
        .node_offset(0)
        .inner_hash_length(HASH_SIZE);
    params
}

fn chunk_params() -> Params {
    let mut params = common_params();
    params.node_depth(0);
    params
}

fn parent_params() -> Params {
    let mut params = common_params();
    params.node_depth(1);
    params
}

fn hash_chunk(chunk: &[u8], finalization: Finalization) -> Hash {
    let mut params = chunk_params();
    if let Root = finalization {
        params.last_node(true);
    }
    params.hash(chunk)
}

fn hash_parent(left: &Hash, right: &Hash, finalization: Finalization) -> Hash {
    let mut state = parent_params().to_state();
    if let Root = finalization {
        state.set_last_node(true);
    }
    state.update(left.as_bytes());
    state.update(right.as_bytes());
    state.finalize()
}

fn hash_eight_chunk_subtree(
    chunk0: &[u8],
    chunk1: &[u8],
    chunk2: &[u8],
    chunk3: &[u8],
    chunk4: &[u8],
    chunk5: &[u8],
    chunk6: &[u8],
    chunk7: &[u8],
    finalization: Finalization,
) -> Hash {
    // This relies on the fact that finalize_hash does nothing for non-root nodes.
    let params = chunk_params();
    let mut jobs = [
        blake2s_simd::many::HashManyJob::new(&params, chunk0),
        blake2s_simd::many::HashManyJob::new(&params, chunk1),
        blake2s_simd::many::HashManyJob::new(&params, chunk2),
        blake2s_simd::many::HashManyJob::new(&params, chunk3),
        blake2s_simd::many::HashManyJob::new(&params, chunk4),
        blake2s_simd::many::HashManyJob::new(&params, chunk5),
        blake2s_simd::many::HashManyJob::new(&params, chunk6),
        blake2s_simd::many::HashManyJob::new(&params, chunk7),
    ];
    blake2s_simd::many::hash_many(jobs.iter_mut());

    let double0 = hash_parent(&jobs[0].to_hash(), &jobs[1].to_hash(), NotRoot);
    let double1 = hash_parent(&jobs[2].to_hash(), &jobs[3].to_hash(), NotRoot);
    let double2 = hash_parent(&jobs[4].to_hash(), &jobs[5].to_hash(), NotRoot);
    let double3 = hash_parent(&jobs[6].to_hash(), &jobs[7].to_hash(), NotRoot);

    let quad0 = hash_parent(&double0, &double1, NotRoot);
    let quad1 = hash_parent(&double2, &double3, NotRoot);

    hash_parent(&quad0, &quad1, finalization)
}

// This is the current standard Bao function. Note that this repo only contains large benchmarks,
// so there's no serial fallback here for short inputs.
fn bao_standard_recurse(input: &[u8], finalization: Finalization) -> Hash {
    if input.len() <= CHUNK_SIZE {
        return hash_chunk(input, finalization);
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 8 * CHUNK_SIZE {
        return hash_eight_chunk_subtree(
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[4 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[5 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[6 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[7 * CHUNK_SIZE..][..CHUNK_SIZE],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len()));
    let (left_hash, right_hash) = join(
        || bao_standard_recurse(left, NotRoot),
        || bao_standard_recurse(right, NotRoot),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_standard(input: &[u8]) -> Hash {
    bao_standard_recurse(input, Root)
}

type HashVec = ArrayVec<[Hash; blake2s_simd::many::MAX_DEGREE]>;
type JobsVec<'a> = ArrayVec<[blake2s_simd::many::HashManyJob<'a>; blake2s_simd::many::MAX_DEGREE]>;

// Another approach to standard Bao. This implementation uses SIMD even when
// hashing parent nodes, to further cut down on overhead without changing the
// output.
fn bao_parallel_parents_recurse(input: &[u8], degree: usize, out: &mut HashVec) {
    // The top level handles the root (and therefore the single chunk case).
    debug_assert!(input.len() > CHUNK_SIZE);

    if input.len() <= degree * CHUNK_SIZE {
        let chunk_params = chunk_params();
        let mut jobs = JobsVec::new();
        for chunk in input.chunks(CHUNK_SIZE) {
            jobs.push(HashManyJob::new(&chunk_params, chunk));
        }
        blake2s_simd::many::hash_many(jobs.iter_mut());
        for job in &jobs {
            out.push(job.to_hash());
        }
        return;
    }

    let (left_input, right_input) = input.split_at(left_len(input.len()));
    let mut left_out = HashVec::new();
    let mut right_out = HashVec::new();
    join(
        || bao_parallel_parents_recurse(left_input, degree, &mut left_out),
        || bao_parallel_parents_recurse(right_input, degree, &mut right_out),
    );
    let mut parents_array = [0; HASH_SIZE * blake2s_simd::many::MAX_DEGREE * 2];
    let mut parents = parents_array.chunks_exact_mut(2 * HASH_SIZE);
    let mut parents_count = 0;
    let mut left_pairs = left_out.chunks_exact(2);
    let mut right_pairs = right_out.chunks_exact(2);
    for (pair, parent) in left_pairs.by_ref().zip(&mut parents) {
        parent[0..HASH_SIZE].copy_from_slice(pair[0].as_bytes());
        parent[HASH_SIZE..2 * HASH_SIZE].copy_from_slice(pair[1].as_bytes());
        parents_count += 1;
    }
    for (pair, parent) in right_pairs.by_ref().zip(&mut parents) {
        parent[0..HASH_SIZE].copy_from_slice(pair[0].as_bytes());
        parent[HASH_SIZE..2 * HASH_SIZE].copy_from_slice(pair[1].as_bytes());
        parents_count += 1;
    }
    let parent_params = parent_params();
    let mut jobs = JobsVec::new();
    for parent in parents_array
        .chunks_exact(2 * HASH_SIZE)
        .take(parents_count)
    {
        jobs.push(HashManyJob::new(&parent_params, parent));
    }
    blake2s_simd::many::hash_many(jobs.iter_mut());
    for job in &jobs {
        out.push(job.to_hash());
    }
    // Raise any remaining right children to the level above. The left side
    // cannot have a remainder.
    debug_assert_eq!(0, left_pairs.remainder().len());
    for hash in right_pairs.remainder() {
        out.push(hash.clone());
    }
}

pub fn bao_parallel_parents(input: &[u8]) -> blake2s_simd::Hash {
    // Handle the single chunk case explicitly.
    if input.len() <= CHUNK_SIZE {
        return hash_chunk(input, Root);
    }
    let mut out = HashVec::new();
    bao_parallel_parents_recurse(input, blake2s_simd::many::degree(), &mut out);
    debug_assert!(out.len() > 1);
    loop {
        if out.len() == 2 {
            return hash_parent(&out[0], &out[1], Root);
        }
        let mut new_out = HashVec::new();
        let mut pairs = out.chunks_exact(2);
        for pair in &mut pairs {
            new_out.push(hash_parent(&pair[0], &pair[1], NotRoot));
        }
        for hash in pairs.remainder() {
            new_out.push(hash.clone());
        }
        out = new_out;
    }
}

// Modified Bao using LARGE_CHUNK_SIZE. This provides a reference point for
// extremely low parent node overhead, though it probably wouldn't be practical
// to use such large chunks in the standard.
fn bao_large_chunks_recurse(input: &[u8], finalization: Finalization) -> Hash {
    if input.len() <= LARGE_CHUNK_SIZE {
        return hash_chunk(input, finalization);
    }
    // Special case: If the input is exactly 8 chunks, hashing those 8
    // chunks in parallel with SIMD is more efficient than going one by one.
    if input.len() == 8 * LARGE_CHUNK_SIZE {
        return hash_eight_chunk_subtree(
            &input[0 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[1 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[2 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[3 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[0 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[1 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[2 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[3 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len()));
    let (left_hash, right_hash) = join(
        || bao_large_chunks_recurse(left, NotRoot),
        || bao_large_chunks_recurse(right, NotRoot),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_large_chunks(input: &[u8]) -> blake2s_simd::Hash {
    bao_large_chunks_recurse(input, Root)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_standard() {
        let input = vec![0; 1_000_000];
        let expected = "c298c4fb54c75e48ea92a210aa071888a6ada44968d116064269204f3e96bfb9";
        let hash = bao_standard(&input);
        assert_eq!(expected, &*hash.to_hex());
    }

    #[test]
    fn test_parallel_parents() {
        let input = vec![0; 1_000_000];
        let expected = "c298c4fb54c75e48ea92a210aa071888a6ada44968d116064269204f3e96bfb9";
        let hash = bao_parallel_parents(&input);
        assert_eq!(expected, &*hash.to_hex());
    }
}
