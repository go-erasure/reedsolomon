package reedsolomon

import "github.com/go-simd/gf65536"

// galMul writes dst = src * coeff over GF(2^16), shards being big-endian uint16
// words. The region kernels (AVX2/SSSE3, NEON, RVV, LSX, VSX, VX, and a portable
// fallback) live in github.com/go-simd/gf65536, which uses the same reduction
// polynomial as this package (0x1100B). f is unused; it is kept so the call
// sites and the scalar oracle in galois_generic.go keep one signature.
func galMul(f *GF16, dst, src []byte, coeff uint16) {
	gf65536.MulSlice(dst, src, coeff)
}

// galMulAdd accumulates dst ^= src * coeff over GF(2^16); see galMul.
func galMulAdd(f *GF16, dst, src []byte, coeff uint16) {
	gf65536.MulAddSlice(dst, src, coeff)
}

// galMulLE and galMulAddLE are galMul and galMulAdd for shards of little-endian
// words, the layout of PAR2 recovery slices; they run the same kernels.
func galMulLE(f *GF16, dst, src []byte, coeff uint16) {
	gf65536.MulSliceLE(dst, src, coeff)
}

func galMulAddLE(f *GF16, dst, src []byte, coeff uint16) {
	gf65536.MulAddSliceLE(dst, src, coeff)
}
