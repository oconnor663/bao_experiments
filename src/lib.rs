use arrayref::{array_mut_ref, array_ref};
use arrayvec::{ArrayString, ArrayVec};
use blake2s_simd::{
    many::{HashManyJob, MAX_DEGREE as MAX_SIMD_DEGREE},
    Params,
};
use rand::seq::SliceRandom;
use rand::RngCore;
use std::fmt;

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

// In my testing, 8 seems to be the best for throughput.
const NARY: usize = 8;

#[derive(Clone, Copy)]
pub struct Hash {
    bytes: [u8; HASH_SIZE],
}

impl Hash {
    /// Create a new `Hash` from an array of bytes.
    pub fn new(bytes: [u8; HASH_SIZE]) -> Self {
        Self { bytes }
    }

    /// Convert the `Hash` to a byte array. Note that the array type doesn't provide constant time
    /// equality.
    pub fn as_bytes(&self) -> &[u8; HASH_SIZE] {
        &self.bytes
    }

    /// Convert the `Hash` to a lowercase hexadecimal
    /// [`ArrayString`](https://docs.rs/arrayvec/0.4/arrayvec/struct.ArrayString.html).
    pub fn to_hex(&self) -> ArrayString<[u8; 2 * HASH_SIZE]> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.bytes.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }
}

/// This implementation is constant time.
impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.bytes[..], &other.bytes[..])
    }
}

/// This implementation is constant time, if the slice length is `HASH_SIZE`.
impl PartialEq<[u8]> for Hash {
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.bytes[..], other)
    }
}

impl Eq for Hash {}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

impl From<blake2s_simd::Hash> for Hash {
    fn from(hash: blake2s_simd::Hash) -> Hash {
        Hash {
            bytes: *array_ref!(hash.as_bytes(), 0, HASH_SIZE),
        }
    }
}

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
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rng);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

fn chunk_params(finalization: Finalization) -> Params {
    let mut params = common_params();
    params.node_depth(0);
    if Root == finalization {
        params.last_node(true);
    }
    params
}

fn parent_params(finalization: Finalization) -> Params {
    let mut params = common_params();
    params.node_depth(1);
    if Root == finalization {
        params.last_node(true);
    }
    params
}

fn hash_chunk(chunk: &[u8], finalization: Finalization) -> Hash {
    chunk_params(finalization).hash(chunk).into()
}

fn hash_parent(left: &Hash, right: &Hash, finalization: Finalization) -> Hash {
    let mut state = parent_params(finalization).to_state();
    state.update(left.as_bytes());
    state.update(right.as_bytes());
    state.finalize().into()
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
    let params = chunk_params(NotRoot);
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

    let double0 = hash_parent(
        &jobs[0].to_hash().into(),
        &jobs[1].to_hash().into(),
        NotRoot,
    );
    let double1 = hash_parent(
        &jobs[2].to_hash().into(),
        &jobs[3].to_hash().into(),
        NotRoot,
    );
    let double2 = hash_parent(
        &jobs[4].to_hash().into(),
        &jobs[5].to_hash().into(),
        NotRoot,
    );
    let double3 = hash_parent(
        &jobs[6].to_hash().into(),
        &jobs[7].to_hash().into(),
        NotRoot,
    );

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
    // Special case: If the input is exactly 8 chunks, hashing those 8 chunks
    // in parallel with SIMD is more efficient than going one by one.
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

fn large_left_len(content_len: usize) -> usize {
    debug_assert!(content_len > LARGE_CHUNK_SIZE);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / LARGE_CHUNK_SIZE;
    largest_power_of_two_leq(full_chunks) * LARGE_CHUNK_SIZE
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
    let (left, right) = input.split_at(large_left_len(input.len()));
    let (left_hash, right_hash) = join(
        || bao_large_chunks_recurse(left, NotRoot),
        || bao_large_chunks_recurse(right, NotRoot),
    );
    hash_parent(&left_hash, &right_hash, finalization)
}

pub fn bao_large_chunks(input: &[u8]) -> Hash {
    bao_large_chunks_recurse(input, Root)
}

type JobsVec<'a> = ArrayVec<[HashManyJob<'a>; MAX_SIMD_DEGREE]>;

// Do one round of constructing and hashing parent hashes.
fn bao_parallel_parents_hash_parents(
    children: &[u8],
    finalization: Finalization,
    out: &mut [u8],
) -> usize {
    debug_assert_eq!(children.len() % HASH_SIZE, 0);
    // finalization=Root means that the current set of children will form the
    // top of the tree, but we can't actually apply Root finalization until we
    // get to the very top node.
    let actual_finalization = if children.len() == 2 * HASH_SIZE {
        finalization
    } else {
        NotRoot
    };
    let params = parent_params(actual_finalization);
    let mut jobs: ArrayVec<[HashManyJob; MAX_SIMD_DEGREE]> = ArrayVec::new();
    let mut pairs = children.chunks_exact(2 * HASH_SIZE);
    for pair in &mut pairs {
        jobs.push(HashManyJob::new(&params, pair))
    }
    blake2s_simd::many::hash_many(&mut jobs);
    let mut out_hashes = out.chunks_exact_mut(HASH_SIZE);
    let mut outputs = 0;
    for (job, out_hash) in jobs.iter().zip(&mut out_hashes) {
        *array_mut_ref!(out_hash, 0, HASH_SIZE) = *job.to_hash().as_array();
        outputs += 1;
    }
    // The leftover child case.
    let leftover = pairs.remainder();
    if leftover.len() == HASH_SIZE {
        if let Some(out_hash) = out_hashes.next() {
            *array_mut_ref!(out_hash, 0, HASH_SIZE) = *array_ref!(leftover, 0, HASH_SIZE);
            outputs += 1;
        }
    }
    outputs
}

// Another approach to standard Bao. This implementation uses SIMD even when
// hashing parent nodes, to further cut down on overhead without changing the
// output.
fn bao_parallel_parents_recurse(
    input: &[u8],
    finalization: Finalization,
    simd_degree: usize,
    out: &mut [u8],
) -> usize {
    // The top level handles the one chunk case.
    debug_assert!(input.len() > 0);

    if input.len() <= simd_degree * CHUNK_SIZE {
        // Because the top level handles the one chunk case, chunk hashing is
        // never Root.
        let chunk_params = chunk_params(NotRoot);
        let mut jobs = JobsVec::new();
        for chunk in input.chunks(CHUNK_SIZE) {
            jobs.push(HashManyJob::new(&chunk_params, chunk));
        }
        blake2s_simd::many::hash_many(jobs.iter_mut());
        for (job, dest) in jobs.iter_mut().zip(out.chunks_exact_mut(HASH_SIZE)) {
            *array_mut_ref!(dest, 0, HASH_SIZE) = *job.to_hash().as_array();
        }
        return jobs.len();
    }

    let (left_input, right_input) = input.split_at(left_len(input.len()));
    let mut child_out_array = [0; 2 * MAX_SIMD_DEGREE * HASH_SIZE];
    let (left_out, right_out) = child_out_array.split_at_mut(simd_degree * HASH_SIZE);
    let (left_n, right_n) = join(
        || bao_parallel_parents_recurse(left_input, NotRoot, simd_degree, left_out),
        || bao_parallel_parents_recurse(right_input, NotRoot, simd_degree, right_out),
    );
    // This asserts that the left_out slice was filled, which means all the
    // child hashes are laid out contiguously.
    debug_assert_eq!(simd_degree, left_n, "left subtree always full");
    let num_children = left_n + right_n;
    let children_slice = &child_out_array[..num_children * HASH_SIZE];
    bao_parallel_parents_hash_parents(children_slice, finalization, out)
}

pub fn bao_parallel_parents(input: &[u8]) -> Hash {
    // Handle the single chunk case explicitly.
    if input.len() <= CHUNK_SIZE {
        return hash_chunk(input, Root);
    }
    let simd_degree = blake2s_simd::many::degree();
    let mut children_array = [0; MAX_SIMD_DEGREE * HASH_SIZE];
    let mut num_children =
        bao_parallel_parents_recurse(input, Root, simd_degree, &mut children_array);
    if simd_degree == 1 {
        debug_assert_eq!(num_children, 1);
    } else {
        debug_assert!(num_children > 1);
    }
    // Now we need to combine child_hashes into parent nodes until we're left
    // with the single root hash, then return that.
    loop {
        if num_children == 1 {
            return Hash {
                bytes: *array_ref!(children_array, 0, HASH_SIZE),
            };
        }
        let mut out_array = [0; MAX_SIMD_DEGREE * HASH_SIZE / 2];
        let children_slice = &children_array[..num_children * HASH_SIZE];
        let out_n = bao_parallel_parents_hash_parents(children_slice, Root, &mut out_array);
        children_array[..out_n * HASH_SIZE].copy_from_slice(&out_array[..out_n * HASH_SIZE]);
        num_children = out_n;
    }
}

// Do one round of constructing and hashing parent hashes. The rule for
// leftover children is that if there's exactly one leftover child, it gets
// raised to the level above, but any larger number of leftover children get
// combined into a partial parent node. Most of the time this will be called
// with simd_degree*tree_degree children, if there's enough input, but it also
// gets reused in a loop at the root level to join everything into the root
// hash.
fn bao_nary_hash_parents(children: &[u8], finalization: Finalization, out: &mut [u8]) -> usize {
    // finalization=Root means that the current set of children will form the
    // top of the tree, but we can't actually apply Root finalization until we
    // get to the very top node.
    debug_assert_eq!(children.len() % HASH_SIZE, 0);
    let actual_finalization = if children.len() / HASH_SIZE <= NARY {
        finalization
    } else {
        NotRoot
    };
    let params = parent_params(actual_finalization);
    let mut jobs: ArrayVec<[HashManyJob; MAX_SIMD_DEGREE]> = ArrayVec::new();
    let mut leftover_child: Option<&[u8; HASH_SIZE]> = None;
    let mut groups = 0;
    for child_group in children.chunks(NARY * HASH_SIZE) {
        if child_group.len() == HASH_SIZE {
            // The single leftover child case.
            leftover_child = Some(array_ref!(child_group, 0, HASH_SIZE))
        } else {
            jobs.push(HashManyJob::new(&params, child_group))
        }
        groups += 1;
    }
    blake2s_simd::many::hash_many(&mut jobs);
    let mut out_hashes = out.chunks_exact_mut(HASH_SIZE);
    for (job, out_hash) in jobs.iter().zip(&mut out_hashes) {
        *array_mut_ref!(out_hash, 0, HASH_SIZE) = *job.to_hash().as_array();
    }
    // The single leftover child case again.
    if let (Some(child), Some(out_hash)) = (leftover_child, out_hashes.next()) {
        out_hash.copy_from_slice(child);
    }
    groups
}

// Returns the number of hashes written to `out`. The caller handles the single
// chunk case (including the empty input), so finalization applies only if
// simd_degree == 1.
fn bao_nary_recurse(
    input: &[u8],
    finalization: Finalization,
    simd_degree: usize,
    out: &mut [u8],
) -> usize {
    // The top level handles the one chunk case.
    debug_assert!(input.len() > 0);

    // If we've reached the number of chunks we want to hash at once, do that.
    if input.len() <= simd_degree * CHUNK_SIZE {
        // Because the top level handles the one chunk case, chunk hashing is
        // never Root.
        let chunk_params = chunk_params(NotRoot);
        let mut jobs = JobsVec::new();
        for chunk in input.chunks(CHUNK_SIZE) {
            jobs.push(HashManyJob::new(&chunk_params, chunk));
        }
        blake2s_simd::many::hash_many(jobs.iter_mut());
        for (job, dest) in jobs.iter_mut().zip(out.chunks_exact_mut(HASH_SIZE)) {
            *array_mut_ref!(dest, 0, HASH_SIZE) = *job.to_hash().as_array();
        }
        return jobs.len();
    }

    // Otherwise, split in two (not NARY!) and recurse. What we want to do is
    // accumulate simd_degree*NARY child hashes, so that we can take full
    // advantage of SIMD to hash NARY parent nodes at the same time. If we get
    // too few (that is, half or less of what we wanted), we just copy what we
    // got directly to the out buffer and let our caller accumulate more.
    //
    // The expected behavior here depends on the value of NARY. For NARY=2, no
    // recursive call will ever short-circuit to the caller. For NARY=4, every
    // other recursive call going down the stack will short-circuit. For
    // NARY=8, every third call, and so one.
    //
    // An alternative to this behavior would be to do a NARY-way split rather
    // than always doing a 2-way split, maybe avoiding some extra copies and
    // stack frames. The downside of that approach is that these larger splits
    // aren't guaranteed to make good use of SIMD when simd_degree isn't an
    // even power of NARY. For example, if simd_degree=8 and NARY=4, consider
    // what happens to a 9-chunk input. 9 is bigger than 8, so we compute the
    // size of each child (the largest power of 4 less than the input) and
    // split the input into 4-4-1 chunks, recursing 3 ways (not having enough
    // chunks to make a 4th child). But now none of the branches are big enough
    // to take advantage of 8-way SIMD. What we want to happen in this case is
    // for the chunks to get split 8-1 for individual hashing, and *then* to be
    // combined 4-4-1 into parents in the caller.
    //
    // Splitting 2 ways takes full advantage of SIMD automatically, assuming
    // simd_degree is always a power of 2. It also makes the recursion a lot
    // easier to write, because at any higher degree the number of recursive
    // calls we make isn't constant (in an incomplete tree with NARY > 2, not
    // every parent node is full).
    debug_assert_eq!(simd_degree.count_ones(), 1, "power of two");
    debug_assert_eq!(MAX_SIMD_DEGREE.count_ones(), 1, "power of two");
    debug_assert_eq!(NARY.count_ones(), 1, "power of two");
    let (left_input, right_input) = input.split_at(left_len(input.len()));
    let mut children_array = [0; NARY * MAX_SIMD_DEGREE * HASH_SIZE];
    let children_wanted = NARY * simd_degree;
    let out_slice_len = children_wanted / 2 * HASH_SIZE;
    debug_assert!(out_slice_len <= children_array.len() / 2);
    let (left_child_out, right_child_out) = children_array.split_at_mut(out_slice_len);
    let (left_n, right_n) = rayon::join(
        || bao_nary_recurse(left_input, NotRoot, simd_degree, left_child_out),
        || bao_nary_recurse(right_input, NotRoot, simd_degree, right_child_out),
    );
    let num_children = left_n + right_n;

    // As described above, if we got half or less of what we wanted, just give
    // everything to the caller.
    if num_children <= children_wanted / 2 {
        let (left_parent_out, right_parent_out) = out.split_at_mut(left_n * HASH_SIZE);
        left_parent_out.copy_from_slice(&left_child_out[..left_n * HASH_SIZE]);
        right_parent_out[..right_n * HASH_SIZE]
            .copy_from_slice(&right_child_out[..right_n * HASH_SIZE]);
        return num_children;
    }

    // If we didn't short-circuit above for having too few children, then we
    // need to join parent nodes. In this case, we know the left side filled
    // the left_child_out slice, so all the child hashes are arranged
    // contiguously in children_array. Note that if simd_degree=1, this might
    // finalize, and the root caller must pass in finalization=Root in that
    // case.
    debug_assert_eq!(left_n, children_wanted / 2, "left subtree complete");
    let children_slice = &children_array[..num_children * HASH_SIZE];
    bao_nary_hash_parents(children_slice, finalization, out)
}

// Note that a real nary design would change the value of the fanout BLAKE2
// parameter. But because this is just a performance experiment, we don't
// bother.
pub fn bao_nary(input: &[u8]) -> Hash {
    // Assert the invariants here at the top. Hopefully this helps the
    // optimizer elide some bounds checks in the recursive calls, but I haven't
    // verified this.
    let simd_degree = blake2s_simd::many::degree();
    assert!(simd_degree <= MAX_SIMD_DEGREE);

    // Handle the single chunk case explicitly.
    if input.len() <= CHUNK_SIZE {
        return hash_chunk(input, Root);
    }

    // If simd_degree=1, the recursive call will take care of finalization.
    // Otherwise we'll do it below.
    let finalization = if simd_degree == 1 { Root } else { NotRoot };
    let mut children_array = [0; NARY * MAX_SIMD_DEGREE * HASH_SIZE];
    let mut num_children = bao_nary_recurse(input, finalization, simd_degree, &mut children_array);
    if simd_degree == 1 {
        debug_assert_eq!(num_children, 1);
    } else {
        debug_assert!(num_children > 1);
    }

    // Now we need to combine child_hashes into parent nodes until we're left
    // with the single root hash, then return that.
    loop {
        if num_children == 1 {
            return Hash {
                bytes: *array_ref!(children_array, 0, HASH_SIZE),
            };
        }
        let mut out_array = [0; MAX_SIMD_DEGREE * HASH_SIZE];
        let children_slice = &children_array[..num_children * HASH_SIZE];
        let out_n = bao_nary_hash_parents(children_slice, Root, &mut out_array);
        children_array[..out_n * HASH_SIZE].copy_from_slice(&out_array[..out_n * HASH_SIZE]);
        num_children = out_n;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_standard() {
        let input = b"";
        let expected = "99fa3a0ee4b435ff17157e205f091cac3938e82335e9684446e513ea1c3b698a";
        let hash = bao_standard(input);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; CHUNK_SIZE];
        let expected = "930f49df68777515ba0891aa7ece3918070517c1ae65ad8b39ec7108d96ebce6";
        let hash = bao_standard(&input);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; CHUNK_SIZE + 1];
        let expected = "d414be8f1ac545c71fcbe46a28fe5924f00111d0cca8828a4dfa94e3b72ffb1a";
        let hash = bao_standard(&input);
        assert_eq!(expected, &*hash.to_hex());

        let input = vec![0; 1_000_000];
        let expected = "c298c4fb54c75e48ea92a210aa071888a6ada44968d116064269204f3e96bfb9";
        let hash = bao_standard(&input);
        assert_eq!(expected, &*hash.to_hex());
    }

    const CASES: &[usize] = &[0, 1, CHUNK_SIZE, CHUNK_SIZE + 1, 1_000_000];

    #[test]
    fn test_parallel_parents() {
        for &len in CASES {
            eprintln!("case {}", len);
            let input = vec![0; len];
            let expected = bao_standard(&input).to_hex();
            let hash = bao_parallel_parents(&input);
            assert_eq!(expected, hash.to_hex());
        }
    }

    fn nary_parent_hash(children: &[Hash], finalization: Finalization) -> Hash {
        let mut state = parent_params(finalization).to_state();
        for child in children {
            state.update(child.as_bytes());
        }
        state.finalize().into()
    }

    #[test]
    fn test_nary_8() {
        assert_eq!(NARY, 8, "value of NARY has changed");

        let chunk = &[0; CHUNK_SIZE];
        let chunk_hash = hash_chunk(chunk, NotRoot);

        // The 8 chunk case.
        let eight_expected_root = nary_parent_hash(&[chunk_hash; 8], Root);
        assert_eq!(eight_expected_root, bao_nary(&[0; 8 * CHUNK_SIZE]));

        // The 9 chunk case.
        let eight_chunks_hash = nary_parent_hash(&[chunk_hash; 8], NotRoot);
        let nine_expected_root = nary_parent_hash(&[eight_chunks_hash, chunk_hash], Root);
        assert_eq!(nine_expected_root, bao_nary(&[0; 9 * CHUNK_SIZE]));

        // The 10 chunk case.
        let two_chunks_hash = nary_parent_hash(&[chunk_hash; 2], NotRoot);
        let ten_expected_root = nary_parent_hash(&[eight_chunks_hash, two_chunks_hash], Root);
        assert_eq!(ten_expected_root, bao_nary(&[0; 10 * CHUNK_SIZE]));
    }
}
