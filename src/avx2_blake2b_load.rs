use arrayref::mut_array_refs;
use byteorder::{ByteOrder, LittleEndian};
use rand::prelude::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const BLAKE2B_BLOCKBYTES: usize = 128;

#[inline(always)]
unsafe fn load_256_from_4xu64(word1: u64, word2: u64, word3: u64, word4: u64) -> __m256i {
    _mm256_setr_epi64x(word1 as i64, word2 as i64, word3 as i64, word4 as i64)
}

#[inline(always)]
unsafe fn load_128_from_4xu32(word1: u32, word2: u32, word3: u32, word4: u32) -> __m128i {
    _mm_setr_epi32(word1 as i32, word2 as i32, word3 as i32, word4 as i32)
}

#[inline(always)]
unsafe fn load_one_msg_vec_simple(
    msg0: &[u8; BLAKE2B_BLOCKBYTES],
    msg1: &[u8; BLAKE2B_BLOCKBYTES],
    msg2: &[u8; BLAKE2B_BLOCKBYTES],
    msg3: &[u8; BLAKE2B_BLOCKBYTES],
    i: usize,
) -> __m256i {
    load_256_from_4xu64(
        LittleEndian::read_u64(&msg0[8 * i..]),
        LittleEndian::read_u64(&msg1[8 * i..]),
        LittleEndian::read_u64(&msg2[8 * i..]),
        LittleEndian::read_u64(&msg3[8 * i..]),
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_simple(
    msg0: &[u8; BLAKE2B_BLOCKBYTES],
    msg1: &[u8; BLAKE2B_BLOCKBYTES],
    msg2: &[u8; BLAKE2B_BLOCKBYTES],
    msg3: &[u8; BLAKE2B_BLOCKBYTES],
) -> [__m256i; 16] {
    [
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 0),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 1),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 2),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 3),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 4),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 5),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 6),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 7),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 8),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 9),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 10),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 11),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 12),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 13),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 14),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, 15),
    ]
}

#[inline(always)]
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256(a, b, 0x20),
        _mm256_permute2x128_si256(a, b, 0x31),
    )
}

#[inline(always)]
unsafe fn interleave_vecs(
    vec_a: __m256i,
    vec_b: __m256i,
    vec_c: __m256i,
    vec_d: __m256i,
) -> [__m256i; 4] {
    // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is 11/33.
    let ab_02 = _mm256_unpacklo_epi64(vec_a, vec_b);
    let ab_13 = _mm256_unpackhi_epi64(vec_a, vec_b);
    let cd_02 = _mm256_unpacklo_epi64(vec_c, vec_d);
    let cd_13 = _mm256_unpackhi_epi64(vec_c, vec_d);

    // Interleave 128-bit lanes.
    let (abcd_0, abcd_2) = interleave128(ab_02, cd_02);
    let (abcd_1, abcd_3) = interleave128(ab_13, cd_13);

    [abcd_0, abcd_1, abcd_2, abcd_3]
}

#[inline(always)]
unsafe fn load_4x256(msg: &[u8; BLAKE2B_BLOCKBYTES]) -> (__m256i, __m256i, __m256i, __m256i) {
    (
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(0)),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(1)),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(2)),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(3)),
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_interleave(
    msg_a: &[u8; BLAKE2B_BLOCKBYTES],
    msg_b: &[u8; BLAKE2B_BLOCKBYTES],
    msg_c: &[u8; BLAKE2B_BLOCKBYTES],
    msg_d: &[u8; BLAKE2B_BLOCKBYTES],
) -> [__m256i; 16] {
    let (a0, a1, a2, a3) = load_4x256(msg_a);
    let (b0, b1, b2, b3) = load_4x256(msg_b);
    let (c0, c1, c2, c3) = load_4x256(msg_c);
    let (d0, d1, d2, d3) = load_4x256(msg_d);

    let interleaved0 = interleave_vecs(a0, b0, c0, d0);
    let interleaved1 = interleave_vecs(a1, b1, c1, d1);
    let interleaved2 = interleave_vecs(a2, b2, c2, d2);
    let interleaved3 = interleave_vecs(a3, b3, c3, d3);
    [
        interleaved0[0],
        interleaved0[1],
        interleaved0[2],
        interleaved0[3],
        interleaved1[0],
        interleaved1[1],
        interleaved1[2],
        interleaved1[3],
        interleaved2[0],
        interleaved2[1],
        interleaved2[2],
        interleaved2[3],
        interleaved3[0],
        interleaved3[1],
        interleaved3[2],
        interleaved3[3],
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn gather_from_blocks(blocks: &[u8; 4 * BLAKE2B_BLOCKBYTES]) -> [__m256i; 16] {
    let indexes = load_128_from_4xu32(
        0 * BLAKE2B_BLOCKBYTES as u32,
        1 * BLAKE2B_BLOCKBYTES as u32,
        2 * BLAKE2B_BLOCKBYTES as u32,
        3 * BLAKE2B_BLOCKBYTES as u32,
    );
    [
        // Safety note: I don't believe VPGATHERDD has alignment requirements.
        _mm256_i32gather_epi64(blocks.as_ptr().add(0) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(8) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(16) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(24) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(32) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(40) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(48) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(56) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(64) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(72) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(80) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(88) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(96) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(104) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(112) as *const i64, indexes, 1),
        _mm256_i32gather_epi64(blocks.as_ptr().add(120) as *const i64, indexes, 1),
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_gather(
    msg_0: &[u8; BLAKE2B_BLOCKBYTES],
    msg_1: &[u8; BLAKE2B_BLOCKBYTES],
    msg_2: &[u8; BLAKE2B_BLOCKBYTES],
    msg_3: &[u8; BLAKE2B_BLOCKBYTES],
) -> [__m256i; 16] {
    let mut buf = [0u8; 4 * BLAKE2B_BLOCKBYTES];
    {
        let refs = mut_array_refs!(
            &mut buf,
            BLAKE2B_BLOCKBYTES,
            BLAKE2B_BLOCKBYTES,
            BLAKE2B_BLOCKBYTES,
            BLAKE2B_BLOCKBYTES
        );
        *refs.0 = *msg_0;
        *refs.1 = *msg_1;
        *refs.2 = *msg_2;
        *refs.3 = *msg_3;
    }
    gather_from_blocks(&buf)
}

pub fn random_block() -> [u8; BLAKE2B_BLOCKBYTES] {
    let mut bytes = [0; BLAKE2B_BLOCKBYTES];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
}

pub fn random_4_blocks() -> [u8; 4 * BLAKE2B_BLOCKBYTES] {
    let mut bytes = [0; 4 * BLAKE2B_BLOCKBYTES];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
}

#[cfg(test)]
mod test {
    use super::*;
    use arrayref::array_refs;
    use std::mem;

    #[cfg(test)]
    fn cast_out(x: __m256i) -> [u64; 4] {
        unsafe { mem::transmute(x) }
    }

    #[test]
    fn test_interleave128() {
        unsafe {
            let a = load_256_from_4xu64(10, 11, 12, 13);
            let b = load_256_from_4xu64(20, 21, 22, 23);

            let expected_a = load_256_from_4xu64(10, 11, 20, 21);
            let expected_b = load_256_from_4xu64(12, 13, 22, 23);

            let (out_a, out_b) = interleave128(a, b);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
        }
    }

    #[test]
    fn test_load_4x256() {
        unsafe {
            let input: [u64; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let input_bytes: [u8; BLAKE2B_BLOCKBYTES] = mem::transmute(input);
            let (out_a, out_b, out_c, out_d) = load_4x256(&input_bytes);

            let expected_a = load_256_from_4xu64(0, 1, 2, 3);
            let expected_b = load_256_from_4xu64(4, 5, 6, 7);
            let expected_c = load_256_from_4xu64(8, 9, 10, 11);
            let expected_d = load_256_from_4xu64(12, 13, 14, 15);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
            assert_eq!(cast_out(expected_c), cast_out(out_c));
            assert_eq!(cast_out(expected_d), cast_out(out_d));
        }
    }

    #[test]
    fn test_interleave_vecs() {
        unsafe {
            let vec_a = load_256_from_4xu64(0x00, 0x01, 0x02, 0x03);
            let vec_b = load_256_from_4xu64(0x10, 0x11, 0x12, 0x13);
            let vec_c = load_256_from_4xu64(0x20, 0x21, 0x22, 0x23);
            let vec_d = load_256_from_4xu64(0x30, 0x31, 0x32, 0x33);

            let expected_a = load_256_from_4xu64(0x00, 0x10, 0x20, 0x30);
            let expected_b = load_256_from_4xu64(0x01, 0x11, 0x21, 0x31);
            let expected_c = load_256_from_4xu64(0x02, 0x12, 0x22, 0x32);
            let expected_d = load_256_from_4xu64(0x03, 0x13, 0x23, 0x33);

            let [out_a, out_b, out_c, out_d] = interleave_vecs(vec_a, vec_b, vec_c, vec_d);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
            assert_eq!(cast_out(expected_c), cast_out(out_c));
            assert_eq!(cast_out(expected_d), cast_out(out_d));
        }
    }

    #[test]
    fn test_load_msg_implementations() {
        unsafe {
            let mut input = [0u64; 4 * 16];
            for i in 0..input.len() {
                input[i] = i as u64;
            }
            let input_bytes: [u8; 4 * BLAKE2B_BLOCKBYTES] = mem::transmute(input);
            let blocks = array_refs!(
                &input_bytes,
                BLAKE2B_BLOCKBYTES,
                BLAKE2B_BLOCKBYTES,
                BLAKE2B_BLOCKBYTES,
                BLAKE2B_BLOCKBYTES
            );

            let expected_vecs = load_msg_vecs_simple(blocks.0, blocks.1, blocks.2, blocks.3);

            let interleave_vecs = load_msg_vecs_interleave(blocks.0, blocks.1, blocks.2, blocks.3);

            let gather_vecs = load_msg_vecs_gather(blocks.0, blocks.1, blocks.2, blocks.3);

            for i in 0..expected_vecs.len() {
                assert_eq!(cast_out(expected_vecs[i]), cast_out(interleave_vecs[i]));
                assert_eq!(cast_out(expected_vecs[i]), cast_out(gather_vecs[i]));
            }
        }
    }
}
