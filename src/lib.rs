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

// This is the current standard Bao function. Note that this repo only contains large benchmarks,
// so there's no non-rayon version of this for short inputs.
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

// A variant of standard Bao using BLAKE2s.
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
    four_ary_parent_hash_blake2b(
        &chunk_hashes[0],
        &chunk_hashes[1],
        &chunk_hashes[2],
        &chunk_hashes[3],
        finalization,
    )
}

// A variant of standard Bao sticking with BLAKE2b but using a 4-ary tree. Here we don't even
// bother to handle trees that aren't a power of 4 number of chunks, but of course we'd need to
// define what to do there if we went with a 4-ary tree.
pub fn hash_recurse_rayon_blake2b_4ary(
    input: &[u8],
    finalization: Finalization,
) -> blake2b_simd::Hash {
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
    let ((child0, child1), (child2, child3)) = rayon::join(
        || {
            rayon::join(
                || hash_recurse_rayon_blake2b_4ary(&input[0 * quarter..][..quarter], NotRoot),
                || hash_recurse_rayon_blake2b_4ary(&input[1 * quarter..][..quarter], NotRoot),
            )
        },
        || {
            rayon::join(
                || hash_recurse_rayon_blake2b_4ary(&input[2 * quarter..][..quarter], NotRoot),
                || hash_recurse_rayon_blake2b_4ary(&input[3 * quarter..][..quarter], NotRoot),
            )
        },
    );
    four_ary_parent_hash_blake2b(&child0, &child1, &child2, &child3, finalization)
}

// Another approach to standard Bao. This implementation makes sure to use 4-way SIMD even in
// hashing parent nodes, to further cut down on overhead without changing the output. Again we
// don't even bother to handle trees that aren't a power of 4 number of chunks, because we're just
// benchmarking the best case.
pub fn hash_recurse_rayon_blake2b_parallel_parents_recurse(
    input: &[u8],
) -> [blake2b_simd::Hash; 4] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (4 * CHUNK_SIZE));

    if input.len() == 4 * CHUNK_SIZE {
        let mut state0 = new_chunk_state_blake2b();
        let mut state1 = new_chunk_state_blake2b();
        let mut state2 = new_chunk_state_blake2b();
        let mut state3 = new_chunk_state_blake2b();
        blake2b_simd::update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
        );
        return blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
    }

    let (left_children, right_children) = rayon::join(
        || hash_recurse_rayon_blake2b_parallel_parents_recurse(&input[..input.len() / 2]),
        || hash_recurse_rayon_blake2b_parallel_parents_recurse(&input[input.len() / 2..]),
    );
    let mut state0 = new_parent_state_blake2b();
    let mut state1 = new_parent_state_blake2b();
    let mut state2 = new_parent_state_blake2b();
    let mut state3 = new_parent_state_blake2b();
    blake2b_simd::update4(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &left_children[0].as_bytes(),
        &left_children[2].as_bytes(),
        &right_children[0].as_bytes(),
        &right_children[2].as_bytes(),
    );
    blake2b_simd::update4(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &left_children[1].as_bytes(),
        &left_children[3].as_bytes(),
        &right_children[1].as_bytes(),
        &right_children[3].as_bytes(),
    );
    blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
}

pub fn hash_recurse_rayon_blake2b_parallel_parents(input: &[u8]) -> blake2b_simd::Hash {
    let children = hash_recurse_rayon_blake2b_parallel_parents_recurse(input);
    let left_hash = parent_hash_blake2b(&children[0], &children[1], NotRoot);
    let right_hash = parent_hash_blake2b(&children[2], &children[3], NotRoot);
    parent_hash_blake2b(&left_hash, &right_hash, Root(input.len() as u64))
}

// A variant that takes advantage of *both* 4-way parent hashing and a 4-ary tree.
pub fn hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(
    input: &[u8],
) -> [blake2b_simd::Hash; 4] {
    // A real version of this algorithm would of course need to handle uneven inputs.
    assert!(input.len() > 0);
    assert_eq!(0, input.len() % (4 * CHUNK_SIZE));

    if input.len() == 4 * CHUNK_SIZE {
        let mut state0 = new_chunk_state_blake2b();
        let mut state1 = new_chunk_state_blake2b();
        let mut state2 = new_chunk_state_blake2b();
        let mut state3 = new_chunk_state_blake2b();
        blake2b_simd::update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input[0 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[1 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[2 * CHUNK_SIZE..][..CHUNK_SIZE],
            &input[3 * CHUNK_SIZE..][..CHUNK_SIZE],
        );
        return blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
    }

    let quarter = input.len() / 4;
    let ((children0, children1), (children2, children3)) = rayon::join(
        || {
            rayon::join(
                || {
                    hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(
                        &input[0 * quarter..][..quarter],
                    )
                },
                || {
                    hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(
                        &input[1 * quarter..][..quarter],
                    )
                },
            )
        },
        || {
            rayon::join(
                || {
                    hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(
                        &input[2 * quarter..][..quarter],
                    )
                },
                || {
                    hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(
                        &input[3 * quarter..][..quarter],
                    )
                },
            )
        },
    );

    let mut state0 = new_parent_state_blake2b();
    let mut state1 = new_parent_state_blake2b();
    let mut state2 = new_parent_state_blake2b();
    let mut state3 = new_parent_state_blake2b();
    for i in 0..4 {
        blake2b_simd::update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &children0[i].as_bytes(),
            &children1[i].as_bytes(),
            &children2[i].as_bytes(),
            &children3[i].as_bytes(),
        );
    }
    blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
}

pub fn hash_recurse_rayon_blake2b_4ary_parallel_parents(input: &[u8]) -> blake2b_simd::Hash {
    let children = hash_recurse_rayon_blake2b_4ary_parallel_parents_recurse(input);
    four_ary_parent_hash_blake2b(
        &children[0],
        &children[1],
        &children[2],
        &children[3],
        Root(input.len() as u64),
    )
}

// We don't have test vectors for the BLAKE2s or the 4-way BLAKE2b implementations, but we can at
// least test the two implementations above that produce standard output.
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_standard() {
        let input = vec![0; 1_000_000];
        let expected = "9a6b54aa320c7ad83f7367aa7206b265a5f86f2ec306e0e108843695c8474311";
        let hash = hash_recurse_rayon_blake2b(&input, Root(input.len() as u64));
        assert_eq!(expected, &*hash.to_hex());
    }

    #[test]
    fn test_standard_parallel_parents() {
        let input = vec![0; 1 << 24];
        let expected = "99ca8f9f6a14d792bc33425268739f28c4a24f817eb92431101e20886a102c1d";
        let hash = hash_recurse_rayon_blake2b_parallel_parents(&input);
        assert_eq!(expected, &*hash.to_hex());
    }

    // Likewise, we can at least make sure that the two 4-ary implementations produce the same
    // output as each other.
    #[test]
    fn test_4ary_implementations() {
        let input = vec![0; 1 << 24];
        let hash1 = hash_recurse_rayon_blake2b_4ary(&input, Root(input.len() as u64));
        let hash2 = hash_recurse_rayon_blake2b_4ary_parallel_parents(&input);
        assert_eq!(hash1, hash2);
    }
}
