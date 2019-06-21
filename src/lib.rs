use arrayref::{array_mut_ref, array_ref};
use arrayvec::ArrayVec;
use blake2s_simd::{
    many::{HashManyJob, MAX_DEGREE},
    Hash, Params,
};
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

pub const HASH_SIZE: usize = 32;
pub const SMALL_CHUNK_SIZE: usize = 1024;
pub const CHUNK_SIZE: usize = 4096;
pub const LARGE_CHUNK_SIZE: usize = 65536;
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

fn left_len(content_len: usize, chunk_size: usize) -> usize {
    debug_assert!(content_len > chunk_size);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / chunk_size;
    largest_power_of_two_leq(full_chunks) * chunk_size
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

fn hash_parent_buf(buf: &[u8; 2 * HASH_SIZE], finalization: Finalization) -> Hash {
    let mut params = parent_params();
    if let Root = finalization {
        params.last_node(true);
    }
    params.hash(buf)
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
        HashManyJob::new(&params, chunk0),
        HashManyJob::new(&params, chunk1),
        HashManyJob::new(&params, chunk2),
        HashManyJob::new(&params, chunk3),
        HashManyJob::new(&params, chunk4),
        HashManyJob::new(&params, chunk5),
        HashManyJob::new(&params, chunk6),
        HashManyJob::new(&params, chunk7),
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

// This is the standard Bao function, with SIMD parallelism enabled only for chunks.
fn bao_basic_recurse(input: &[u8], finalization: Finalization, chunk_size: usize) -> Hash {
    if input.len() <= chunk_size {
        return hash_chunk(input, finalization);
    }
    // Special case: If the input is exactly 8 chunks, hashing those 8 chunks
    // in parallel with SIMD is more efficient than going one by one.
    if input.len() == 8 * chunk_size {
        return hash_eight_chunk_subtree(
            &input[0 * chunk_size..][..chunk_size],
            &input[1 * chunk_size..][..chunk_size],
            &input[2 * chunk_size..][..chunk_size],
            &input[3 * chunk_size..][..chunk_size],
            &input[4 * chunk_size..][..chunk_size],
            &input[5 * chunk_size..][..chunk_size],
            &input[6 * chunk_size..][..chunk_size],
            &input[7 * chunk_size..][..chunk_size],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len(), chunk_size));
    let (left_hash, right_hash) = join(
        || bao_basic_recurse(left, NotRoot, chunk_size),
        || bao_basic_recurse(right, NotRoot, chunk_size),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_basic(input: &[u8], chunk_size: usize) -> Hash {
    bao_basic_recurse(input, Root, chunk_size)
}

const OUT_BUF_LEN: usize = 2 * MAX_DEGREE * HASH_SIZE;
type JobsVec<'a> = ArrayVec<[HashManyJob<'a>; MAX_DEGREE]>;

// The same function as bao_basic, but this time we enable parallelism for the
// parents too, all the way up the tree.
fn bao_parallel_parents_recurse(
    input: &[u8],
    out: &mut [u8; OUT_BUF_LEN],
    chunk_size: usize,
) -> usize {
    // The top level handles the empty case.
    debug_assert!(input.len() > 0);

    if input.len() <= MAX_DEGREE * chunk_size {
        let chunk_params = chunk_params();
        let mut jobs = JobsVec::new();
        for chunk in input.chunks(chunk_size) {
            jobs.push(HashManyJob::new(&chunk_params, chunk));
        }
        blake2s_simd::many::hash_many(jobs.iter_mut());
        for (job, dest) in jobs.iter_mut().zip(out.chunks_exact_mut(HASH_SIZE)) {
            job.write_output(array_mut_ref!(dest, 0, HASH_SIZE));
        }
        return jobs.len();
    }

    let (left_input, right_input) = input.split_at(left_len(input.len(), chunk_size));
    let mut left_out = [0; OUT_BUF_LEN];
    let mut right_out = [0; OUT_BUF_LEN];
    let (left_n, right_n) = join(
        || bao_parallel_parents_recurse(left_input, &mut left_out, chunk_size),
        || bao_parallel_parents_recurse(right_input, &mut right_out, chunk_size),
    );
    debug_assert_eq!(MAX_DEGREE, left_n, "left subtree always full");
    let parent_params = parent_params();
    let mut jobs = JobsVec::new();
    let left_parent_bufs = left_out.chunks_exact(2 * HASH_SIZE).take(left_n / 2);
    let right_parent_bufs = right_out.chunks_exact(2 * HASH_SIZE).take(right_n / 2);
    for in_buf in left_parent_bufs.chain(right_parent_bufs) {
        jobs.push(HashManyJob::new(&parent_params, in_buf));
    }
    blake2s_simd::many::hash_many(jobs.iter_mut());
    for (job, out_buf) in jobs.iter().zip(out.chunks_exact_mut(HASH_SIZE)) {
        job.write_output(array_mut_ref!(out_buf, 0, HASH_SIZE));
    }

    // If there's a right child left over, copy it to the level above.
    let num_outputs = MAX_DEGREE / 2 + (right_n + 1) / 2;
    if right_n % 2 == 1 {
        let last_child = &right_out[(right_n - 1) * HASH_SIZE..][..HASH_SIZE];
        let last_output = &mut out[(num_outputs - 1) * HASH_SIZE..][..HASH_SIZE];
        last_output.copy_from_slice(last_child);
    }

    num_outputs
}

pub fn bao_parallel_parents(input: &[u8], chunk_size: usize) -> Hash {
    // Handle the single chunk case explicitly.
    if input.len() <= chunk_size {
        return hash_chunk(input, Root);
    }
    let mut out = [0; OUT_BUF_LEN];
    let mut n = bao_parallel_parents_recurse(input, &mut out, chunk_size);
    debug_assert!(n > 1);
    loop {
        if n == 2 {
            return hash_parent_buf(array_ref!(out, 0, 2 * HASH_SIZE), Root);
        }
        let mut i = 0;
        while i < n {
            let offset = i * HASH_SIZE;
            if i + 1 < n {
                // Join two child hashes into one parent hash and write the
                // result back in place.
                let parent_hash = hash_parent_buf(array_ref!(&out, offset, 2 * HASH_SIZE), NotRoot);
                array_mut_ref!(&mut out, offset / 2, HASH_SIZE)
                    .copy_from_slice(parent_hash.as_bytes());
            } else {
                // Copy the leftover child hash.
                let last_hash = *array_ref!(&out, offset, HASH_SIZE);
                *array_mut_ref!(&mut out, offset / 2, HASH_SIZE) = last_hash;
            }
            i += 2;
        }
        n = n / 2 + (n % 2) as usize;
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
            &input[4 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[5 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[6 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            &input[7 * LARGE_CHUNK_SIZE..][..LARGE_CHUNK_SIZE],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len(), LARGE_CHUNK_SIZE));
    let (left_hash, right_hash) = join(
        || bao_large_chunks_recurse(left, NotRoot),
        || bao_large_chunks_recurse(right, NotRoot),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_large_chunks(input: &[u8]) -> Hash {
    bao_large_chunks_recurse(input, Root)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic() {
        let input = b"";
        let expected = "99fa3a0ee4b435ff17157e205f091cac3938e82335e9684446e513ea1c3b698a";
        let hash = bao_basic(input, CHUNK_SIZE);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; CHUNK_SIZE];
        let expected = "930f49df68777515ba0891aa7ece3918070517c1ae65ad8b39ec7108d96ebce6";
        let hash = bao_basic(&input, CHUNK_SIZE);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; CHUNK_SIZE + 1];
        let expected = "d414be8f1ac545c71fcbe46a28fe5924f00111d0cca8828a4dfa94e3b72ffb1a";
        let hash = bao_basic(&input, CHUNK_SIZE);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; 1_000_000];
        let expected = "c298c4fb54c75e48ea92a210aa071888a6ada44968d116064269204f3e96bfb9";
        let hash = bao_basic(&input, CHUNK_SIZE);
        assert_eq!(expected, &*hash.to_hex());
    }

    #[test]
    fn test_parallel_parents() {
        for &len in &[0, 1, CHUNK_SIZE, CHUNK_SIZE + 1, 1_000_000] {
            eprintln!("case {}", len);
            let input = vec![0; len];
            let expected = bao_basic(&input, CHUNK_SIZE).to_hex();
            let hash = bao_parallel_parents(&input, CHUNK_SIZE);
            assert_eq!(expected, hash.to_hex());
        }
    }
}
