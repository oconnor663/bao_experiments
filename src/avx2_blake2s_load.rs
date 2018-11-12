use arrayref::mut_array_refs;
use byteorder::{ByteOrder, LittleEndian};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const BLAKE2S_BLOCKBYTES: usize = 64;

#[inline(always)]
unsafe fn load_256_from_8xu32(
    word1: u32,
    word2: u32,
    word3: u32,
    word4: u32,
    word5: u32,
    word6: u32,
    word7: u32,
    word8: u32,
) -> __m256i {
    _mm256_setr_epi32(
        word1 as i32,
        word2 as i32,
        word3 as i32,
        word4 as i32,
        word5 as i32,
        word6 as i32,
        word7 as i32,
        word8 as i32,
    )
}

#[inline(always)]
unsafe fn load_one_msg_vec_simple(
    msg0: &[u8; BLAKE2S_BLOCKBYTES],
    msg1: &[u8; BLAKE2S_BLOCKBYTES],
    msg2: &[u8; BLAKE2S_BLOCKBYTES],
    msg3: &[u8; BLAKE2S_BLOCKBYTES],
    msg4: &[u8; BLAKE2S_BLOCKBYTES],
    msg5: &[u8; BLAKE2S_BLOCKBYTES],
    msg6: &[u8; BLAKE2S_BLOCKBYTES],
    msg7: &[u8; BLAKE2S_BLOCKBYTES],
    i: usize,
) -> __m256i {
    load_256_from_8xu32(
        LittleEndian::read_u32(&msg0[4 * i..]),
        LittleEndian::read_u32(&msg1[4 * i..]),
        LittleEndian::read_u32(&msg2[4 * i..]),
        LittleEndian::read_u32(&msg3[4 * i..]),
        LittleEndian::read_u32(&msg4[4 * i..]),
        LittleEndian::read_u32(&msg5[4 * i..]),
        LittleEndian::read_u32(&msg6[4 * i..]),
        LittleEndian::read_u32(&msg7[4 * i..]),
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_simple(
    msg0: &[u8; BLAKE2S_BLOCKBYTES],
    msg1: &[u8; BLAKE2S_BLOCKBYTES],
    msg2: &[u8; BLAKE2S_BLOCKBYTES],
    msg3: &[u8; BLAKE2S_BLOCKBYTES],
    msg4: &[u8; BLAKE2S_BLOCKBYTES],
    msg5: &[u8; BLAKE2S_BLOCKBYTES],
    msg6: &[u8; BLAKE2S_BLOCKBYTES],
    msg7: &[u8; BLAKE2S_BLOCKBYTES],
) -> [__m256i; 16] {
    [
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 0),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 1),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 2),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 3),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 4),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 5),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 6),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 7),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 8),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 9),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 10),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 11),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 12),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 13),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 14),
        load_one_msg_vec_simple(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 15),
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
unsafe fn load_2x256(msg: &[u8; BLAKE2S_BLOCKBYTES]) -> (__m256i, __m256i) {
    (
        _mm256_loadu_si256(msg.as_ptr() as *const __m256i),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(1)),
    )
}

#[inline(always)]
unsafe fn interleave_vecs(
    vec_a: __m256i,
    vec_b: __m256i,
    vec_c: __m256i,
    vec_d: __m256i,
    vec_e: __m256i,
    vec_f: __m256i,
    vec_g: __m256i,
    vec_h: __m256i,
) -> [__m256i; 8] {
    // Interleave 32-bit lanes. The low unpack is lanes 00/11/44/55, and the high is 22/33/66/77.
    let ab_0145 = _mm256_unpacklo_epi32(vec_a, vec_b);
    let ab_2367 = _mm256_unpackhi_epi32(vec_a, vec_b);
    let cd_0145 = _mm256_unpacklo_epi32(vec_c, vec_d);
    let cd_2367 = _mm256_unpackhi_epi32(vec_c, vec_d);
    let ef_0145 = _mm256_unpacklo_epi32(vec_e, vec_f);
    let ef_2367 = _mm256_unpackhi_epi32(vec_e, vec_f);
    let gh_0145 = _mm256_unpacklo_epi32(vec_g, vec_h);
    let gh_2367 = _mm256_unpackhi_epi32(vec_g, vec_h);

    // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is 11/33.
    let abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
    let abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
    let abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
    let abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
    let efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
    let efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
    let efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
    let efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

    // Interleave 128-bit lanes.
    let (abcdefg_0, abcdefg_4) = interleave128(abcd_04, efgh_04);
    let (abcdefg_1, abcdefg_5) = interleave128(abcd_15, efgh_15);
    let (abcdefg_2, abcdefg_6) = interleave128(abcd_26, efgh_26);
    let (abcdefg_3, abcdefg_7) = interleave128(abcd_37, efgh_37);

    [
        abcdefg_0, abcdefg_1, abcdefg_2, abcdefg_3, abcdefg_4, abcdefg_5, abcdefg_6, abcdefg_7,
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_interleave(
    msg_a: &[u8; BLAKE2S_BLOCKBYTES],
    msg_b: &[u8; BLAKE2S_BLOCKBYTES],
    msg_c: &[u8; BLAKE2S_BLOCKBYTES],
    msg_d: &[u8; BLAKE2S_BLOCKBYTES],
    msg_e: &[u8; BLAKE2S_BLOCKBYTES],
    msg_f: &[u8; BLAKE2S_BLOCKBYTES],
    msg_g: &[u8; BLAKE2S_BLOCKBYTES],
    msg_h: &[u8; BLAKE2S_BLOCKBYTES],
) -> [__m256i; 16] {
    let (front_a, back_a) = load_2x256(msg_a);
    let (front_b, back_b) = load_2x256(msg_b);
    let (front_c, back_c) = load_2x256(msg_c);
    let (front_d, back_d) = load_2x256(msg_d);
    let (front_e, back_e) = load_2x256(msg_e);
    let (front_f, back_f) = load_2x256(msg_f);
    let (front_g, back_g) = load_2x256(msg_g);
    let (front_h, back_h) = load_2x256(msg_h);

    let front_interleaved = interleave_vecs(
        front_a, front_b, front_c, front_d, front_e, front_f, front_g, front_h,
    );
    let back_interleaved = interleave_vecs(
        back_a, back_b, back_c, back_d, back_e, back_f, back_g, back_h,
    );

    [
        front_interleaved[0],
        front_interleaved[1],
        front_interleaved[2],
        front_interleaved[3],
        front_interleaved[4],
        front_interleaved[5],
        front_interleaved[6],
        front_interleaved[7],
        back_interleaved[0],
        back_interleaved[1],
        back_interleaved[2],
        back_interleaved[3],
        back_interleaved[4],
        back_interleaved[5],
        back_interleaved[6],
        back_interleaved[7],
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn gather_from_blocks(blocks: &[u8; 8 * BLAKE2S_BLOCKBYTES]) -> [__m256i; 16] {
    let indexes = load_256_from_8xu32(
        0 * BLAKE2S_BLOCKBYTES as u32,
        1 * BLAKE2S_BLOCKBYTES as u32,
        2 * BLAKE2S_BLOCKBYTES as u32,
        3 * BLAKE2S_BLOCKBYTES as u32,
        4 * BLAKE2S_BLOCKBYTES as u32,
        5 * BLAKE2S_BLOCKBYTES as u32,
        6 * BLAKE2S_BLOCKBYTES as u32,
        7 * BLAKE2S_BLOCKBYTES as u32,
    );
    [
        // Safety note: I don't believe VPGATHERDD has alignment requirements.
        _mm256_i32gather_epi32(blocks.as_ptr().add(0) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(4) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(8) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(12) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(16) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(20) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(24) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(28) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(32) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(36) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(40) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(44) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(48) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(52) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(56) as *const i32, indexes, 1),
        _mm256_i32gather_epi32(blocks.as_ptr().add(60) as *const i32, indexes, 1),
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn load_msg_vecs_gather(
    msg_0: &[u8; BLAKE2S_BLOCKBYTES],
    msg_1: &[u8; BLAKE2S_BLOCKBYTES],
    msg_2: &[u8; BLAKE2S_BLOCKBYTES],
    msg_3: &[u8; BLAKE2S_BLOCKBYTES],
    msg_4: &[u8; BLAKE2S_BLOCKBYTES],
    msg_5: &[u8; BLAKE2S_BLOCKBYTES],
    msg_6: &[u8; BLAKE2S_BLOCKBYTES],
    msg_7: &[u8; BLAKE2S_BLOCKBYTES],
) -> [__m256i; 16] {
    let mut buf = [0u8; 8 * BLAKE2S_BLOCKBYTES];
    {
        let refs = mut_array_refs!(
            &mut buf,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES,
            BLAKE2S_BLOCKBYTES
        );
        *refs.0 = *msg_0;
        *refs.1 = *msg_1;
        *refs.2 = *msg_2;
        *refs.3 = *msg_3;
        *refs.4 = *msg_4;
        *refs.5 = *msg_5;
        *refs.6 = *msg_6;
        *refs.7 = *msg_7;
    }
    gather_from_blocks(&buf)
}

#[cfg(test)]
mod test {
    use super::*;
    use arrayref::array_refs;
    use std::mem;

    #[cfg(test)]
    fn cast_out(x: __m256i) -> [u32; 8] {
        unsafe { mem::transmute(x) }
    }

    #[test]
    fn test_interleave128() {
        unsafe {
            let a = load_256_from_8xu32(10, 11, 12, 13, 14, 15, 16, 17);
            let b = load_256_from_8xu32(20, 21, 22, 23, 24, 25, 26, 27);

            let expected_a = load_256_from_8xu32(10, 11, 12, 13, 20, 21, 22, 23);
            let expected_b = load_256_from_8xu32(14, 15, 16, 17, 24, 25, 26, 27);

            let (out_a, out_b) = interleave128(a, b);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
        }
    }

    #[test]
    fn test_load_2x256() {
        unsafe {
            let input: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let input_bytes: [u8; BLAKE2S_BLOCKBYTES] = mem::transmute(input);
            let (out_a, out_b) = load_2x256(&input_bytes);

            let expected_a = load_256_from_8xu32(0, 0, 1, 0, 2, 0, 3, 0);
            let expected_b = load_256_from_8xu32(4, 0, 5, 0, 6, 0, 7, 0);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
        }
    }

    #[test]
    fn test_interleave_vecs() {
        unsafe {
            let vec_a = load_256_from_8xu32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
            let vec_b = load_256_from_8xu32(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17);
            let vec_c = load_256_from_8xu32(0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27);
            let vec_d = load_256_from_8xu32(0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37);
            let vec_e = load_256_from_8xu32(0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47);
            let vec_f = load_256_from_8xu32(0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57);
            let vec_g = load_256_from_8xu32(0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67);
            let vec_h = load_256_from_8xu32(0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77);

            let expected_a = load_256_from_8xu32(0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70);
            let expected_b = load_256_from_8xu32(0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71);
            let expected_c = load_256_from_8xu32(0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72);
            let expected_d = load_256_from_8xu32(0x03, 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73);
            let expected_e = load_256_from_8xu32(0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74);
            let expected_f = load_256_from_8xu32(0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75);
            let expected_g = load_256_from_8xu32(0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76);
            let expected_h = load_256_from_8xu32(0x07, 0x17, 0x27, 0x37, 0x47, 0x57, 0x67, 0x77);

            let [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h] =
                interleave_vecs(vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h);

            assert_eq!(cast_out(expected_a), cast_out(out_a));
            assert_eq!(cast_out(expected_b), cast_out(out_b));
            assert_eq!(cast_out(expected_c), cast_out(out_c));
            assert_eq!(cast_out(expected_d), cast_out(out_d));
            assert_eq!(cast_out(expected_e), cast_out(out_e));
            assert_eq!(cast_out(expected_f), cast_out(out_f));
            assert_eq!(cast_out(expected_g), cast_out(out_g));
            assert_eq!(cast_out(expected_h), cast_out(out_h));
        }
    }

    #[test]
    fn test_load_msg_implementations() {
        unsafe {
            let mut input = [0u32; 8 * 16];
            for i in 0..input.len() {
                input[i] = i as u32;
            }
            let input_bytes: [u8; 8 * BLAKE2S_BLOCKBYTES] = mem::transmute(input);
            let blocks = array_refs!(
                &input_bytes,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES,
                BLAKE2S_BLOCKBYTES
            );

            let expected_vecs = load_msg_vecs_simple(
                blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
            );

            let interleave_vecs = load_msg_vecs_interleave(
                blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
            );

            let gather_vecs = load_msg_vecs_gather(
                blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
            );

            for i in 0..expected_vecs.len() {
                assert_eq!(cast_out(expected_vecs[i]), cast_out(interleave_vecs[i]));
                assert_eq!(cast_out(expected_vecs[i]), cast_out(gather_vecs[i]));
            }
        }
    }
}
