extern crate blake2b_simd;
extern crate blake2s_simd;
extern crate byteorder;
extern crate rayon;

use byteorder::{ByteOrder, LittleEndian};

const HASH_SIZE: usize = 32;
const HEADER_SIZE: usize = 8;
const CHUNK_SIZE: usize = 4096;

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

fn new_chunk_state_blake2b() -> blake2b_simd::State {
    common_params_blake2b().node_depth(0).to_state()
}

fn new_parent_state_blake2b() -> blake2b_simd::State {
    common_params_blake2b().node_depth(1).to_state()
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

fn new_chunk_state_blake2s() -> blake2s_simd::State {
    common_params_blake2s().node_depth(0).to_state()
}

fn new_parent_state_blake2s() -> blake2s_simd::State {
    common_params_blake2s().node_depth(1).to_state()
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
    let mut state0 = new_chunk_state_blake2b();
    let mut state1 = new_chunk_state_blake2b();
    let mut state2 = new_chunk_state_blake2b();
    let mut state3 = new_chunk_state_blake2b();
    blake2b_simd::update4(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        chunk0,
        chunk1,
        chunk2,
        chunk3,
    );
    let chunk_hashes = blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
    let left_hash = parent_hash_blake2b(&chunk_hashes[0], &chunk_hashes[1], NotRoot);
    let right_hash = parent_hash_blake2b(&chunk_hashes[2], &chunk_hashes[3], NotRoot);
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
    let mut state0 = new_chunk_state_blake2s();
    let mut state1 = new_chunk_state_blake2s();
    let mut state2 = new_chunk_state_blake2s();
    let mut state3 = new_chunk_state_blake2s();
    let mut state4 = new_chunk_state_blake2s();
    let mut state5 = new_chunk_state_blake2s();
    let mut state6 = new_chunk_state_blake2s();
    let mut state7 = new_chunk_state_blake2s();
    blake2s_simd::update8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
        chunk0,
        chunk1,
        chunk2,
        chunk3,
        chunk4,
        chunk5,
        chunk6,
        chunk7,
    );
    let chunk_hashes = blake2s_simd::finalize8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
    );

    let double0 = parent_hash_blake2s(&chunk_hashes[0], &chunk_hashes[1], NotRoot);
    let double1 = parent_hash_blake2s(&chunk_hashes[2], &chunk_hashes[3], NotRoot);
    let double2 = parent_hash_blake2s(&chunk_hashes[4], &chunk_hashes[5], NotRoot);
    let double3 = parent_hash_blake2s(&chunk_hashes[6], &chunk_hashes[7], NotRoot);

    let quad0 = parent_hash_blake2s(&double0, &double1, NotRoot);
    let quad1 = parent_hash_blake2s(&double2, &double3, NotRoot);

    parent_hash_blake2s(&quad0, &quad1, finalization)
}

// This repo only contains large benchmarks, so there's no non-rayon version of this for short
// inputs.
pub fn hash_recurse_rayon_blake2b(input: &[u8], finalization: Finalization) -> blake2b_simd::Hash {
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
    let (left_hash, right_hash) = rayon::join(
        || hash_recurse_rayon_blake2b(left, NotRoot),
        || hash_recurse_rayon_blake2b(right, NotRoot),
    );
    parent_hash_blake2b(&left_hash, &right_hash, finalization)
}

// This repo only contains large benchmarks, so there's no non-rayon version of this for short
// inputs.
pub fn hash_recurse_rayon_blake2s(input: &[u8], finalization: Finalization) -> blake2s_simd::Hash {
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
    let (left_hash, right_hash) = rayon::join(
        || hash_recurse_rayon_blake2s(left, NotRoot),
        || hash_recurse_rayon_blake2s(right, NotRoot),
    );
    parent_hash_blake2s(&left_hash, &right_hash, finalization)
}

// We don't have test vectors for the BLAKE2s or the 4-way BLAKE2b implementations, but we can at
// least test the standard one above.
#[test]
fn test_standard() {
    let input = vec![0; 1_000_000];
    let expected = "9a6b54aa320c7ad83f7367aa7206b265a5f86f2ec306e0e108843695c8474311";
    let hash = hash_recurse_rayon_blake2b(&input, Root(input.len() as u64));
    assert_eq!(expected, &*hash.to_hex());
}
