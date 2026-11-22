package reedsolomon

// This file holds the portable, CGO-free scalar field-region hot loops that
// dominate Encode/Reconstruct/Verify. The galMul/galMulAdd entry points
// (galois.go) delegate to github.com/go-simd/gf65536, which owns the SIMD
// kernels. These loops are an independent log-table implementation: the
// correctness oracle the differential tests (galois_simd_test.go) check
// gf65536 against, bit for bit.
//
// Shards are byte slices interpreted as big-endian uint16 words (the LE
// variants below read little-endian words, the PAR2 layout). src is
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

// galMulLEScalar and galMulAddLEScalar are the little-endian-word oracles for
// galMulLE and galMulAddLE (first byte of each pair is the low byte).
func galMulLEScalar(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		p := f.Mul(coeff, uint16(src[w+1])<<8|uint16(src[w]))
		dst[w] = byte(p)
		dst[w+1] = byte(p >> 8)
	}
}

func galMulAddLEScalar(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		p := f.Mul(coeff, uint16(src[w+1])<<8|uint16(src[w]))
		dst[w] ^= byte(p)
		dst[w+1] ^= byte(p >> 8)
	}
}
