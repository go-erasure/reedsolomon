package reedsolomon

// This file holds the portable, CGO-free scalar field-region hot loops that
// dominate Encode/Reconstruct/Verify. On amd64 (SSSE3) and arm64 (NEON) the
// public galMul/galMulAdd entry points (galois_amd64.go / galois_arm64.go) fold
// the bulk of each region through a go-asmgen SIMD split-table kernel and defer
// only the sub-block tail here; on every other architecture galois_fallback.go
// routes the whole region through these scalar loops. They are therefore both
// the universal fallback AND the correctness oracle the differential tests
// (galois_simd_test.go) check the SIMD kernels against, bit for bit.
//
// Shards are byte slices interpreted as big-endian uint16 words. src is
// consumed two bytes at a time; any trailing odd byte is ignored (callers
// guarantee even-length shards).

// galMulScalar writes dst = src * coeff over GF(2^16), word by word. dst and src
// must have the same length.
func galMulScalar(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		v := uint16(src[w])<<8 | uint16(src[w+1])
		p := f.Mul(coeff, v)
		dst[w] = byte(p >> 8)
		dst[w+1] = byte(p)
	}
}

// galMulAddScalar accumulates dst ^= src * coeff over GF(2^16), word by word.
// This is the multiply-accumulate region kernel used throughout matrix
// application. dst and src must have the same length.
func galMulAddScalar(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		v := uint16(src[w])<<8 | uint16(src[w+1])
		p := f.Mul(coeff, v)
		d := uint16(dst[w])<<8 | uint16(dst[w+1])
		d ^= p
		dst[w] = byte(d >> 8)
		dst[w+1] = byte(d)
	}
}
