use blake2s_simd::Hash;

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2b_load;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2s_load;

fn largest_power_of_two_leq(n: u64) -> u64 {
    ((n / 2) + 1).next_power_of_two()
}

fn left_len(content_len: u64) -> u64 {
    debug_assert!(content_len > CHUNK_SIZE as u64);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / CHUNK_SIZE as u64;
    largest_power_of_two_leq(full_chunks) * CHUNK_SIZE as u64
}

#[derive(Clone, Copy, Debug)]
pub enum Finalization {
    NotRoot,
    Root,
}
use self::Finalization::{NotRoot, Root};

fn common_params() -> blake2s_simd::Params {
    let mut params = blake2s_simd::Params::new();
    params
        .hash_length(HASH_SIZE)
        .fanout(2)
        .max_depth(64)
        .max_leaf_length(CHUNK_SIZE as u32)
        .node_offset(0)
        .inner_hash_length(HASH_SIZE);
    params
}

fn chunk_params() -> blake2s_simd::Params {
    let mut params = common_params();
    params.node_depth(0);
    params
}

fn parent_params() -> blake2s_simd::Params {
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
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
    let (left_hash, right_hash) = join(
        || bao_standard_recurse(left, NotRoot),
        || bao_standard_recurse(right, NotRoot),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_standard(input: &[u8]) -> Hash {
    bao_standard_recurse(input, Root)
}

// Another approach to standard Bao. This implementation makes sure to use
// 8-way SIMD even in hashing parent nodes, to further cut down on overhead
// without changing the output. We don't even bother to handle trees that
// aren't a power of 8 number of chunks, because we're just benchmarking the
// best case.
fn bao_parallel_parents_recurse(input: &[u8]) -> [Hash; 8] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (8 * CHUNK_SIZE));

    if input.len() == 8 * CHUNK_SIZE {
        let params = chunk_params();
        let mut jobs = [
            blake2s_simd::many::HashManyJob::new(&params, &input[0 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[1 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[2 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[3 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[4 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[5 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[6 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2s_simd::many::HashManyJob::new(&params, &input[7 * CHUNK_SIZE..][..CHUNK_SIZE]),
        ];
        blake2s_simd::many::hash_many(jobs.iter_mut());
        return [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
            jobs[4].to_hash(),
            jobs[5].to_hash(),
            jobs[6].to_hash(),
            jobs[7].to_hash(),
        ];
    }

    let (left_children, right_children) = join(
        || bao_parallel_parents_recurse(&input[..input.len() / 2]),
        || bao_parallel_parents_recurse(&input[input.len() / 2..]),
    );
    let mut input0 = [0; 2 * HASH_SIZE];
    input0[..HASH_SIZE].copy_from_slice(left_children[0].as_bytes());
    input0[HASH_SIZE..].copy_from_slice(left_children[1].as_bytes());
    let mut input1 = [0; 2 * HASH_SIZE];
    input1[..HASH_SIZE].copy_from_slice(left_children[2].as_bytes());
    input1[HASH_SIZE..].copy_from_slice(left_children[3].as_bytes());
    let mut input2 = [0; 2 * HASH_SIZE];
    input2[..HASH_SIZE].copy_from_slice(left_children[4].as_bytes());
    input2[HASH_SIZE..].copy_from_slice(left_children[5].as_bytes());
    let mut input3 = [0; 2 * HASH_SIZE];
    input3[..HASH_SIZE].copy_from_slice(left_children[6].as_bytes());
    input3[HASH_SIZE..].copy_from_slice(left_children[7].as_bytes());
    let mut input4 = [0; 2 * HASH_SIZE];
    input4[..HASH_SIZE].copy_from_slice(right_children[0].as_bytes());
    input4[HASH_SIZE..].copy_from_slice(right_children[1].as_bytes());
    let mut input5 = [0; 2 * HASH_SIZE];
    input5[..HASH_SIZE].copy_from_slice(right_children[2].as_bytes());
    input5[HASH_SIZE..].copy_from_slice(right_children[3].as_bytes());
    let mut input6 = [0; 2 * HASH_SIZE];
    input6[..HASH_SIZE].copy_from_slice(right_children[4].as_bytes());
    input6[HASH_SIZE..].copy_from_slice(right_children[5].as_bytes());
    let mut input7 = [0; 2 * HASH_SIZE];
    input7[..HASH_SIZE].copy_from_slice(right_children[6].as_bytes());
    input7[HASH_SIZE..].copy_from_slice(right_children[7].as_bytes());

    let params = parent_params();
    let mut jobs = [
        blake2s_simd::many::HashManyJob::new(&params, &input0),
        blake2s_simd::many::HashManyJob::new(&params, &input1),
        blake2s_simd::many::HashManyJob::new(&params, &input2),
        blake2s_simd::many::HashManyJob::new(&params, &input3),
        blake2s_simd::many::HashManyJob::new(&params, &input4),
        blake2s_simd::many::HashManyJob::new(&params, &input5),
        blake2s_simd::many::HashManyJob::new(&params, &input6),
        blake2s_simd::many::HashManyJob::new(&params, &input7),
    ];
    blake2s_simd::many::hash_many(jobs.iter_mut());
    [
        jobs[0].to_hash(),
        jobs[1].to_hash(),
        jobs[2].to_hash(),
        jobs[3].to_hash(),
        jobs[4].to_hash(),
        jobs[5].to_hash(),
        jobs[6].to_hash(),
        jobs[7].to_hash(),
    ]
}

pub fn bao_parallel_parents(input: &[u8]) -> blake2s_simd::Hash {
    let children = bao_parallel_parents_recurse(input);
    let double0 = hash_parent(&children[0], &children[1], NotRoot);
    let double1 = hash_parent(&children[2], &children[3], NotRoot);
    let double2 = hash_parent(&children[4], &children[5], NotRoot);
    let double3 = hash_parent(&children[6], &children[7], NotRoot);
    let quad0 = hash_parent(&double0, &double1, NotRoot);
    let quad1 = hash_parent(&double2, &double3, NotRoot);
    hash_parent(&quad0, &quad1, Root)
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
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
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
        let input = vec![0; 1 << 24];
        let expected = "a50d86c04ce3dae9060f4721a833a547c30bc9a39f7a5875c78dfcc4f83f6931";
        let hash = bao_parallel_parents(&input);
        assert_eq!(expected, &*hash.to_hex());
    }
}
