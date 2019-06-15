extern crate arrayref;
extern crate blake2b_simd;
extern crate blake2s_simd;
extern crate byteorder;
extern crate rand;
extern crate rayon;

use byteorder::{ByteOrder, LittleEndian};

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
const HEADER_SIZE: usize = 8;
const CHUNK_SIZE: usize = 4096;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2b_load;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_blake2s_load;

fn encode_len(len: u64) -> [u8; HEADER_SIZE] {
    let mut len_bytes = [0; HEADER_SIZE];
    LittleEndian::write_u64(&mut len_bytes, len);
    len_bytes
}

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
    Root(u64),
}
use self::Finalization::{NotRoot, Root};

fn common_params_blake2b() -> blake2b_simd::Params {
    let mut params = blake2b_simd::Params::new();
    params
        .hash_length(HASH_SIZE)
        .fanout(2)
        .max_depth(64)
        .max_leaf_length(CHUNK_SIZE as u32)
        .node_offset(0)
        .inner_hash_length(HASH_SIZE);
    params
}

fn chunk_params_blake2b() -> blake2b_simd::Params {
    let mut params = common_params_blake2b();
    params.node_depth(0);
    params
}

fn parent_params_blake2b() -> blake2b_simd::Params {
    let mut params = common_params_blake2b();
    params.node_depth(1);
    params
}

fn new_chunk_state_blake2b() -> blake2b_simd::State {
    chunk_params_blake2b().to_state()
}

fn new_parent_state_blake2b() -> blake2b_simd::State {
    parent_params_blake2b().to_state()
}

fn finalize_hash_blake2b(
    state: &mut blake2b_simd::State,
    finalization: Finalization,
) -> blake2b_simd::Hash {
    if let Root(root_len) = finalization {
        state.update(&encode_len(root_len));
        state.set_last_node(true);
    }
    state.finalize()
}

fn common_params_blake2s() -> blake2s_simd::Params {
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

fn chunk_params_blake2s() -> blake2s_simd::Params {
    let mut params = common_params_blake2s();
    params.node_depth(0);
    params
}

fn parent_params_blake2s() -> blake2s_simd::Params {
    let mut params = common_params_blake2s();
    params.node_depth(1);
    params
}

fn new_chunk_state_blake2s() -> blake2s_simd::State {
    chunk_params_blake2s().to_state()
}

fn new_parent_state_blake2s() -> blake2s_simd::State {
    parent_params_blake2s().to_state()
}

fn finalize_hash_blake2s(
    state: &mut blake2s_simd::State,
    finalization: Finalization,
) -> blake2s_simd::Hash {
    if let Root(root_len) = finalization {
        state.update(&encode_len(root_len));
        state.set_last_node(true);
    }
    state.finalize()
}

fn hash_chunk_blake2b(chunk: &[u8], finalization: Finalization) -> blake2b_simd::Hash {
    let mut state = new_chunk_state_blake2b();
    state.update(chunk);
    finalize_hash_blake2b(&mut state, finalization)
}

fn hash_chunk_blake2s(chunk: &[u8], finalization: Finalization) -> blake2s_simd::Hash {
    let mut state = new_chunk_state_blake2s();
    state.update(chunk);
    finalize_hash_blake2s(&mut state, finalization)
}

fn parent_hash_blake2b(
    left: &blake2b_simd::Hash,
    right: &blake2b_simd::Hash,
    finalization: Finalization,
) -> blake2b_simd::Hash {
    let mut state = new_parent_state_blake2b();
    state.update(left.as_bytes());
    state.update(right.as_bytes());
    finalize_hash_blake2b(&mut state, finalization)
}

fn parent_hash_blake2s(
    left: &blake2s_simd::Hash,
    right: &blake2s_simd::Hash,
    finalization: Finalization,
) -> blake2s_simd::Hash {
    let mut state = new_parent_state_blake2s();
    state.update(left.as_bytes());
    state.update(right.as_bytes());
    finalize_hash_blake2s(&mut state, finalization)
}

fn hash_four_chunk_subtree_blake2b(
    chunk0: &[u8],
    chunk1: &[u8],
    chunk2: &[u8],
    chunk3: &[u8],
    finalization: Finalization,
) -> blake2b_simd::Hash {
    // This relies on the fact that finalize_hash does nothing for non-root nodes.
    let params = chunk_params_blake2b();
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, chunk0),
        blake2b_simd::many::HashManyJob::new(&params, chunk1),
        blake2b_simd::many::HashManyJob::new(&params, chunk2),
        blake2b_simd::many::HashManyJob::new(&params, chunk3),
    ];
    blake2b_simd::many::hash_many(jobs.iter_mut());
    let left_hash = parent_hash_blake2b(&jobs[0].to_hash(), &jobs[1].to_hash(), NotRoot);
    let right_hash = parent_hash_blake2b(&jobs[2].to_hash(), &jobs[3].to_hash(), NotRoot);
    parent_hash_blake2b(&left_hash, &right_hash, finalization)
}

fn hash_eight_chunk_subtree_blake2s(
    chunk0: &[u8],
    chunk1: &[u8],
    chunk2: &[u8],
    chunk3: &[u8],
    chunk4: &[u8],
    chunk5: &[u8],
    chunk6: &[u8],
    chunk7: &[u8],
    finalization: Finalization,
) -> blake2s_simd::Hash {
    // This relies on the fact that finalize_hash does nothing for non-root nodes.
    let params = chunk_params_blake2s();
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

    let double0 = parent_hash_blake2s(&jobs[0].to_hash(), &jobs[1].to_hash(), NotRoot);
    let double1 = parent_hash_blake2s(&jobs[2].to_hash(), &jobs[3].to_hash(), NotRoot);
    let double2 = parent_hash_blake2s(&jobs[4].to_hash(), &jobs[5].to_hash(), NotRoot);
    let double3 = parent_hash_blake2s(&jobs[6].to_hash(), &jobs[7].to_hash(), NotRoot);

    let quad0 = parent_hash_blake2s(&double0, &double1, NotRoot);
    let quad1 = parent_hash_blake2s(&double2, &double3, NotRoot);

    parent_hash_blake2s(&quad0, &quad1, finalization)
}

// This is the current standard Bao function. Note that this repo only contains large benchmarks,
// so there's no serial fallback here for short inputs.
fn bao_standard_recurse(input: &[u8], finalization: Finalization) -> blake2b_simd::Hash {
    if input.len() <= CHUNK_SIZE {
        return hash_chunk_blake2b(input, finalization);
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 4 * CHUNK_SIZE {
        return hash_four_chunk_subtree_blake2b(
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
    let (left_hash, right_hash) = join(
        || bao_standard_recurse(left, NotRoot),
        || bao_standard_recurse(right, NotRoot),
    );
    parent_hash_blake2b(&left_hash, &right_hash, finalization)
}

pub fn bao_standard(input: &[u8]) -> blake2b_simd::Hash {
    bao_standard_recurse(input, Root(input.len() as u64))
}

// A variant of standard Bao using BLAKE2s.
pub fn bao_blake2s_recurse(input: &[u8], finalization: Finalization) -> blake2s_simd::Hash {
    if input.len() <= CHUNK_SIZE {
        return hash_chunk_blake2s(input, finalization);
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 8 * CHUNK_SIZE {
        return hash_eight_chunk_subtree_blake2s(
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
        || bao_blake2s_recurse(left, NotRoot),
        || bao_blake2s_recurse(right, NotRoot),
    );
    parent_hash_blake2s(&left_hash, &right_hash, finalization)
}

pub fn bao_blake2s(input: &[u8]) -> blake2s_simd::Hash {
    bao_blake2s_recurse(input, Root(input.len() as u64))
}

fn four_ary_parent_hash_blake2b(
    child0: &blake2b_simd::Hash,
    child1: &blake2b_simd::Hash,
    child2: &blake2b_simd::Hash,
    child3: &blake2b_simd::Hash,
    finalization: Finalization,
) -> blake2b_simd::Hash {
    let mut parent = new_parent_state_blake2b();
    parent.update(child0.as_bytes());
    parent.update(child1.as_bytes());
    parent.update(child2.as_bytes());
    parent.update(child3.as_bytes());
    finalize_hash_blake2b(&mut parent, finalization)
}

fn hash_four_chunk_4ary_subtree_blake2b(
    chunk0: &[u8],
    chunk1: &[u8],
    chunk2: &[u8],
    chunk3: &[u8],
    finalization: Finalization,
) -> blake2b_simd::Hash {
    // This relies on the fact that finalize_hash does nothing for non-root nodes.
    let params = chunk_params_blake2b();
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, chunk0),
        blake2b_simd::many::HashManyJob::new(&params, chunk1),
        blake2b_simd::many::HashManyJob::new(&params, chunk2),
        blake2b_simd::many::HashManyJob::new(&params, chunk3),
    ];
    blake2b_simd::many::hash_many(jobs.iter_mut());
    four_ary_parent_hash_blake2b(
        &jobs[0].to_hash(),
        &jobs[1].to_hash(),
        &jobs[2].to_hash(),
        &jobs[3].to_hash(),
        finalization,
    )
}

fn bao_4ary_recurse(input: &[u8], finalization: Finalization) -> blake2b_simd::Hash {
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (4 * CHUNK_SIZE));
    if input.len() == 4 * CHUNK_SIZE {
        return hash_four_chunk_4ary_subtree_blake2b(
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
            finalization,
        );
    }
    let quarter = input.len() / 4;
    let ((child0, child1), (child2, child3)) = join(
        || {
            join(
                || bao_4ary_recurse(&input[0 * quarter..][..quarter], NotRoot),
                || bao_4ary_recurse(&input[1 * quarter..][..quarter], NotRoot),
            )
        },
        || {
            join(
                || bao_4ary_recurse(&input[2 * quarter..][..quarter], NotRoot),
                || bao_4ary_recurse(&input[3 * quarter..][..quarter], NotRoot),
            )
        },
    );
    four_ary_parent_hash_blake2b(&child0, &child1, &child2, &child3, finalization)
}

// A variant of standard Bao sticking with BLAKE2b but using a 4-ary tree. Here we don't even
// bother to handle trees that aren't a power of 4 number of chunks, but of course we'd need to
// define what to do there if we went with a 4-ary tree.
pub fn bao_4ary(input: &[u8]) -> blake2b_simd::Hash {
    bao_4ary_recurse(input, Root(input.len() as u64))
}

// Another approach to standard Bao. This implementation makes sure to use 4-way SIMD even in
// hashing parent nodes, to further cut down on overhead without changing the output. Again we
// don't even bother to handle trees that aren't a power of 4 number of chunks, because we're just
// benchmarking the best case.
fn bao_standard_parallel_parents_recurse(input: &[u8]) -> [blake2b_simd::Hash; 4] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (4 * CHUNK_SIZE));

    if input.len() == 4 * CHUNK_SIZE {
        let params = chunk_params_blake2b();
        let mut jobs = [
            blake2b_simd::many::HashManyJob::new(&params, &input[0 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[1 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[2 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[3 * CHUNK_SIZE..][..CHUNK_SIZE]),
        ];
        blake2b_simd::many::hash_many(jobs.iter_mut());
        return [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ];
    }

    let (left_children, right_children) = join(
        || bao_standard_parallel_parents_recurse(&input[..input.len() / 2]),
        || bao_standard_parallel_parents_recurse(&input[input.len() / 2..]),
    );
    // Note that we can't use hash4_exact here, though we could maybe invent another interface that
    // doesn't assume exactness, and which pays the corresponding overhead.
    let mut input0 = [0; 2 * HASH_SIZE];
    input0[..HASH_SIZE].copy_from_slice(left_children[0].as_bytes());
    input0[HASH_SIZE..].copy_from_slice(left_children[1].as_bytes());
    let mut input1 = [0; 2 * HASH_SIZE];
    input1[..HASH_SIZE].copy_from_slice(left_children[2].as_bytes());
    input1[HASH_SIZE..].copy_from_slice(left_children[3].as_bytes());
    let mut input2 = [0; 2 * HASH_SIZE];
    input2[..HASH_SIZE].copy_from_slice(right_children[0].as_bytes());
    input2[HASH_SIZE..].copy_from_slice(right_children[1].as_bytes());
    let mut input3 = [0; 2 * HASH_SIZE];
    input3[..HASH_SIZE].copy_from_slice(right_children[2].as_bytes());
    input3[HASH_SIZE..].copy_from_slice(right_children[3].as_bytes());

    let params = parent_params_blake2b();
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, &input0),
        blake2b_simd::many::HashManyJob::new(&params, &input1),
        blake2b_simd::many::HashManyJob::new(&params, &input2),
        blake2b_simd::many::HashManyJob::new(&params, &input3),
    ];
    blake2b_simd::many::hash_many(jobs.iter_mut());
    [
        jobs[0].to_hash(),
        jobs[1].to_hash(),
        jobs[2].to_hash(),
        jobs[3].to_hash(),
    ]
}

pub fn bao_standard_parallel_parents(input: &[u8]) -> blake2b_simd::Hash {
    let children = bao_standard_parallel_parents_recurse(input);
    let left_hash = parent_hash_blake2b(&children[0], &children[1], NotRoot);
    let right_hash = parent_hash_blake2b(&children[2], &children[3], NotRoot);
    parent_hash_blake2b(&left_hash, &right_hash, Root(input.len() as u64))
}

// Similar to bao_standard_parallel_parents, but with BLAKE2s.
fn bao_blake2s_parallel_parents_recurse(input: &[u8]) -> [blake2s_simd::Hash; 8] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (8 * CHUNK_SIZE));

    if input.len() == 8 * CHUNK_SIZE {
        let params = chunk_params_blake2s();
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
        || bao_blake2s_parallel_parents_recurse(&input[..input.len() / 2]),
        || bao_blake2s_parallel_parents_recurse(&input[input.len() / 2..]),
    );
    // Note that we can't use hash4_exact here, though we could maybe invent another interface that
    // doesn't assume exactness, and which pays the corresponding overhead.
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

    let params = parent_params_blake2s();
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

pub fn bao_blake2s_parallel_parents(input: &[u8]) -> blake2s_simd::Hash {
    let children = bao_blake2s_parallel_parents_recurse(input);
    let double0 = parent_hash_blake2s(&children[0], &children[1], NotRoot);
    let double1 = parent_hash_blake2s(&children[2], &children[3], NotRoot);
    let double2 = parent_hash_blake2s(&children[4], &children[5], NotRoot);
    let double3 = parent_hash_blake2s(&children[6], &children[7], NotRoot);
    let quad0 = parent_hash_blake2s(&double0, &double1, NotRoot);
    let quad1 = parent_hash_blake2s(&double2, &double3, NotRoot);
    parent_hash_blake2s(&quad0, &quad1, Root(input.len() as u64))
}

// A variant that takes advantage of *both* 4-way parent hashing and a 4-ary tree.
pub fn bao_4ary_parallel_parents_recurse(input: &[u8]) -> [blake2b_simd::Hash; 4] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (4 * CHUNK_SIZE));

    if input.len() == 4 * CHUNK_SIZE {
        let params = chunk_params_blake2b();
        let mut jobs = [
            blake2b_simd::many::HashManyJob::new(&params, &input[0 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[1 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[2 * CHUNK_SIZE..][..CHUNK_SIZE]),
            blake2b_simd::many::HashManyJob::new(&params, &input[3 * CHUNK_SIZE..][..CHUNK_SIZE]),
        ];
        blake2b_simd::many::hash_many(jobs.iter_mut());
        return [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ];
    }

    let quarter = input.len() / 4;
    let ((children0, children1), (children2, children3)) = join(
        || {
            join(
                || bao_4ary_parallel_parents_recurse(&input[0 * quarter..][..quarter]),
                || bao_4ary_parallel_parents_recurse(&input[1 * quarter..][..quarter]),
            )
        },
        || {
            join(
                || bao_4ary_parallel_parents_recurse(&input[2 * quarter..][..quarter]),
                || bao_4ary_parallel_parents_recurse(&input[3 * quarter..][..quarter]),
            )
        },
    );

    let mut parent0 = [0; 4 * HASH_SIZE];
    parent0[0 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children0[0].as_bytes());
    parent0[1 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children0[1].as_bytes());
    parent0[2 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children0[2].as_bytes());
    parent0[3 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children0[3].as_bytes());
    let mut parent1 = [0; 4 * HASH_SIZE];
    parent1[0 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children1[0].as_bytes());
    parent1[1 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children1[1].as_bytes());
    parent1[2 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children1[2].as_bytes());
    parent1[3 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children1[3].as_bytes());
    let mut parent2 = [0; 4 * HASH_SIZE];
    parent2[0 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children2[0].as_bytes());
    parent2[1 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children2[1].as_bytes());
    parent2[2 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children2[2].as_bytes());
    parent2[3 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children2[3].as_bytes());
    let mut parent3 = [0; 4 * HASH_SIZE];
    parent3[0 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children3[0].as_bytes());
    parent3[1 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children3[1].as_bytes());
    parent3[2 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children3[2].as_bytes());
    parent3[3 * HASH_SIZE..][..HASH_SIZE].copy_from_slice(children3[3].as_bytes());
    let params = parent_params_blake2b();
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, &parent0),
        blake2b_simd::many::HashManyJob::new(&params, &parent1),
        blake2b_simd::many::HashManyJob::new(&params, &parent2),
        blake2b_simd::many::HashManyJob::new(&params, &parent3),
    ];
    blake2b_simd::many::hash_many(jobs.iter_mut());
    [
        jobs[0].to_hash(),
        jobs[1].to_hash(),
        jobs[2].to_hash(),
        jobs[3].to_hash(),
    ]
}

pub fn bao_4ary_parallel_parents(input: &[u8]) -> blake2b_simd::Hash {
    let children = bao_4ary_parallel_parents_recurse(input);
    four_ary_parent_hash_blake2b(
        &children[0],
        &children[1],
        &children[2],
        &children[3],
        Root(input.len() as u64),
    )
}

// Modified Bao using a larger chunk size.
fn bao_blake2b_large_chunks_recurse(
    input: &[u8],
    finalization: Finalization,
) -> blake2b_simd::Hash {
    const LARGE_CHUNK: usize = 65536;

    if input.len() <= LARGE_CHUNK {
        return hash_chunk_blake2b(input, finalization);
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 4 * LARGE_CHUNK {
        return hash_four_chunk_subtree_blake2b(
            &input[0 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[1 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[2 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[3 * LARGE_CHUNK..][..LARGE_CHUNK],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
    let (left_hash, right_hash) = join(
        || bao_blake2b_large_chunks_recurse(left, NotRoot),
        || bao_blake2b_large_chunks_recurse(right, NotRoot),
    );
    parent_hash_blake2b(&left_hash, &right_hash, finalization)
}

pub fn bao_blake2b_large_chunks(input: &[u8]) -> blake2b_simd::Hash {
    bao_blake2b_large_chunks_recurse(input, Root(input.len() as u64))
}

// Modified Bao using BLAKE2s and a larger chunk size.
fn bao_blake2s_large_chunks_recurse(
    input: &[u8],
    finalization: Finalization,
) -> blake2s_simd::Hash {
    const LARGE_CHUNK: usize = 65536;

    if input.len() <= LARGE_CHUNK {
        return hash_chunk_blake2s(input, finalization);
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 8 * LARGE_CHUNK {
        return hash_eight_chunk_subtree_blake2s(
            &input[0 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[1 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[2 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[3 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[0 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[1 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[2 * LARGE_CHUNK..][..LARGE_CHUNK],
            &input[3 * LARGE_CHUNK..][..LARGE_CHUNK],
            finalization,
        );
    }
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
    let (left_hash, right_hash) = join(
        || bao_blake2s_large_chunks_recurse(left, NotRoot),
        || bao_blake2s_large_chunks_recurse(right, NotRoot),
    );
    parent_hash_blake2s(&left_hash, &right_hash, finalization)
}

pub fn bao_blake2s_large_chunks(input: &[u8]) -> blake2s_simd::Hash {
    bao_blake2s_large_chunks_recurse(input, Root(input.len() as u64))
}

#[derive(Debug, PartialEq, Eq)]
pub enum Either {
    B(blake2b_simd::Hash),
    S(blake2s_simd::Hash),
}
use Either::*;

impl Either {
    fn as_bytes(&self) -> &[u8] {
        match *self {
            B(ref hash) => hash.as_bytes(),
            S(ref hash) => hash.as_bytes(),
        }
    }
}

fn hash_4_chunks_either(chunk0: &[u8], chunk1: &[u8], chunk2: &[u8], chunk3: &[u8]) -> [Either; 4] {
    let params = chunk_params_blake2b();
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, chunk0),
        blake2b_simd::many::HashManyJob::new(&params, chunk1),
        blake2b_simd::many::HashManyJob::new(&params, chunk2),
        blake2b_simd::many::HashManyJob::new(&params, chunk3),
    ];
    blake2b_simd::many::hash_many(&mut jobs);
    [
        B(jobs[0].to_hash()),
        B(jobs[1].to_hash()),
        B(jobs[2].to_hash()),
        B(jobs[3].to_hash()),
    ]
}

fn parent_hash_either(left: &Either, right: &Either, finalization: Finalization) -> Either {
    let mut parent_state = new_parent_state_blake2s();
    parent_state.update(left.as_bytes());
    parent_state.update(right.as_bytes());
    S(finalize_hash_blake2s(&mut parent_state, finalization))
}

// NOTE: This implementation should be able to take advantage of SSE in the BLAKE2s implementation,
// but that's not currently available.
fn bao_blake2hybrid_recurse(input: &[u8], finalization: Finalization) -> Either {
    if input.len() <= CHUNK_SIZE {
        return B(hash_chunk_blake2b(input, finalization));
    }
    // Special case: If the input is exactly four chunks, hashing those four chunks in parallel
    // with SIMD is more efficient than going one by one.
    if input.len() == 4 * CHUNK_SIZE {
        let children = hash_4_chunks_either(
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
        );
        let double0 = parent_hash_either(&children[0], &children[1], NotRoot);
        let double1 = parent_hash_either(&children[2], &children[3], NotRoot);
        return parent_hash_either(&double0, &double1, finalization);
    }
    let (left, right) = input.split_at(left_len(input.len() as u64) as usize);
    let (left_hash, right_hash) = join(
        || bao_blake2hybrid_recurse(left, NotRoot),
        || bao_blake2hybrid_recurse(right, NotRoot),
    );
    parent_hash_either(&left_hash, &right_hash, finalization)
}

pub fn bao_blake2hybrid(input: &[u8]) -> Either {
    bao_blake2hybrid_recurse(input, Root(input.len() as u64))
}

pub fn bao_blake2hybrid_parallel_parents_recurse(input: &[u8]) -> [Either; 8] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (8 * CHUNK_SIZE));

    if input.len() == 8 * CHUNK_SIZE {
        let children_left = hash_4_chunks_either(
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
        );
        let children_right = hash_4_chunks_either(
            &input[4 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[5 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[6 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[7 * CHUNK_SIZE..][..CHUNK_SIZE],
        );
        match (children_left, children_right) {
            ([h0, h1, h2, h3], [h4, h5, h6, h7]) => return [h0, h1, h2, h3, h4, h5, h6, h7],
        }
    }

    let (left_children, right_children) = join(
        || bao_blake2hybrid_parallel_parents_recurse(&input[..input.len() / 2]),
        || bao_blake2hybrid_parallel_parents_recurse(&input[input.len() / 2..]),
    );
    let mut parent0 = [0; 2 * HASH_SIZE];
    parent0[..HASH_SIZE].copy_from_slice(left_children[0].as_bytes());
    parent0[HASH_SIZE..].copy_from_slice(left_children[1].as_bytes());
    let mut parent1 = [0; 2 * HASH_SIZE];
    parent1[..HASH_SIZE].copy_from_slice(left_children[2].as_bytes());
    parent1[HASH_SIZE..].copy_from_slice(left_children[3].as_bytes());
    let mut parent2 = [0; 2 * HASH_SIZE];
    parent2[..HASH_SIZE].copy_from_slice(left_children[4].as_bytes());
    parent2[HASH_SIZE..].copy_from_slice(left_children[5].as_bytes());
    let mut parent3 = [0; 2 * HASH_SIZE];
    parent3[..HASH_SIZE].copy_from_slice(left_children[6].as_bytes());
    parent3[HASH_SIZE..].copy_from_slice(left_children[7].as_bytes());
    let mut parent4 = [0; 2 * HASH_SIZE];
    parent4[..HASH_SIZE].copy_from_slice(right_children[0].as_bytes());
    parent4[HASH_SIZE..].copy_from_slice(right_children[1].as_bytes());
    let mut parent5 = [0; 2 * HASH_SIZE];
    parent5[..HASH_SIZE].copy_from_slice(right_children[2].as_bytes());
    parent5[HASH_SIZE..].copy_from_slice(right_children[3].as_bytes());
    let mut parent6 = [0; 2 * HASH_SIZE];
    parent6[..HASH_SIZE].copy_from_slice(right_children[4].as_bytes());
    parent6[HASH_SIZE..].copy_from_slice(right_children[5].as_bytes());
    let mut parent7 = [0; 2 * HASH_SIZE];
    parent7[..HASH_SIZE].copy_from_slice(right_children[6].as_bytes());
    parent7[HASH_SIZE..].copy_from_slice(right_children[7].as_bytes());
    let params = parent_params_blake2s();
    let mut jobs = [
        blake2s_simd::many::HashManyJob::new(&params, &parent0),
        blake2s_simd::many::HashManyJob::new(&params, &parent1),
        blake2s_simd::many::HashManyJob::new(&params, &parent2),
        blake2s_simd::many::HashManyJob::new(&params, &parent3),
        blake2s_simd::many::HashManyJob::new(&params, &parent4),
        blake2s_simd::many::HashManyJob::new(&params, &parent5),
        blake2s_simd::many::HashManyJob::new(&params, &parent6),
        blake2s_simd::many::HashManyJob::new(&params, &parent7),
    ];
    blake2s_simd::many::hash_many(jobs.iter_mut());
    [
        S(jobs[0].to_hash()),
        S(jobs[1].to_hash()),
        S(jobs[2].to_hash()),
        S(jobs[3].to_hash()),
        S(jobs[4].to_hash()),
        S(jobs[5].to_hash()),
        S(jobs[6].to_hash()),
        S(jobs[7].to_hash()),
    ]
}

pub fn bao_blake2hybrid_parallel_parents(input: &[u8]) -> Either {
    let children = bao_blake2hybrid_parallel_parents_recurse(input);

    let double0 = parent_hash_either(&children[0], &children[1], NotRoot);
    let double1 = parent_hash_either(&children[2], &children[3], NotRoot);
    let double2 = parent_hash_either(&children[4], &children[5], NotRoot);
    let double3 = parent_hash_either(&children[6], &children[7], NotRoot);

    let quad0 = parent_hash_either(&double0, &double1, NotRoot);
    let quad1 = parent_hash_either(&double2, &double3, NotRoot);

    parent_hash_either(&quad0, &quad1, Root(input.len() as u64))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_standard() {
        let input = vec![0; 1_000_000];
        let expected = "9a6b54aa320c7ad83f7367aa7206b265a5f86f2ec306e0e108843695c8474311";
        let hash = bao_standard(&input);
        assert_eq!(expected, &*hash.to_hex());
    }

    #[test]
    fn test_standard_parallel_parents() {
        let input = vec![0; 1 << 24];
        let expected = "99ca8f9f6a14d792bc33425268739f28c4a24f817eb92431101e20886a102c1d";
        let hash = bao_standard_parallel_parents(&input);
        assert_eq!(expected, &*hash.to_hex());
    }

    // We don't have test vectors for the Bao-BLAKE2s, but we can at least test the two
    // implementations above that produce standard output.
    #[test]
    fn test_blake2s_implementations() {
        let input = vec![0; 1 << 24];
        let hash1 = bao_blake2s(&input);
        let hash2 = bao_blake2s_parallel_parents(&input);
        assert_eq!(hash1, hash2);
    }

    // Likewise, we can at least make sure that the two 4-ary implementations produce the same
    // output as each other.
    #[test]
    fn test_4ary_implementations() {
        let input = vec![0; 1 << 24];
        let hash1 = bao_4ary(&input);
        let hash2 = bao_4ary_parallel_parents(&input);
        assert_eq!(hash1, hash2);
    }

    // Likewise, we can at least make sure that the two hybrid implementations produce the same
    // output as each other.
    #[test]
    fn test_hybrid_implementations() {
        let input = vec![0; 32 * CHUNK_SIZE];
        let hash1 = bao_blake2hybrid(&input);
        let hash2 = bao_blake2hybrid_parallel_parents(&input);
        assert_eq!(hash1, hash2);
    }
}
