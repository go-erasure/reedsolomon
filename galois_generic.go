package reedsolomon

// This file holds the field-region hot loops that dominate Encode/Reconstruct/
// Verify. They are isolated behind clean package-level signatures so a future
// go-asmgen SIMD variant can replace them without touching the Reed-Solomon
// logic. The default build is this pure-Go, CGO-free scalar implementation.
//
// Shards are byte slices interpreted as big-endian uint16 words. src is
// consumed two bytes at a time; any trailing odd byte is ignored (callers
// guarantee even-length shards).

// galMul writes dst = src * coeff over GF(2^16), word by word.
// dst and src must have the same (even) length.
func galMul(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		v := uint16(src[w])<<8 | uint16(src[w+1])
		p := f.Mul(coeff, v)
		dst[w] = byte(p >> 8)
		dst[w+1] = byte(p)
	}
}

// galMulAdd accumulates dst ^= src * coeff over GF(2^16), word by word. This is
// the multiply-accumulate region kernel used throughout matrix application.
// dst and src must have the same (even) length.
func galMulAdd(f *GF16, dst, src []byte, coeff uint16) {
	for w := 0; w+1 < len(src); w += 2 {
		v := uint16(src[w])<<8 | uint16(src[w+1])
		p := f.Mul(coeff, v)
		d := uint16(dst[w])<<8 | uint16(dst[w+1])
		d ^= p
		dst[w] = byte(d >> 8)
		dst[w+1] = byte(d)
	}
}
