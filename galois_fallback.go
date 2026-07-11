//go:build !amd64 && !arm64 && !s390x && !loong64 && !ppc64le && !riscv64

package reedsolomon

// On architectures without a go-asmgen SIMD kernel (wasm, 386, mips, …) the
// region multiply routes entirely through the scalar oracle. amd64 (SSSE3),
// arm64 (NEON), s390x (VPERM), ppc64le (VSX VPERM), riscv64 (RVV vrgather) and
// loong64 (LSX vshuf.b) all have verified split-table kernels; these scalar
// loops are byte-for-byte identical to them by construction (the differential
// test proves it on every arch, under QEMU for the emulated ones), so downstream
// behaviour is unchanged and only the throughput differs.

// galMul writes dst = src * coeff over GF(2^16).
func galMul(f *GF16, dst, src []byte, coeff uint16) {
	galMulScalar(f, dst, src, coeff)
}

// galMulAdd accumulates dst ^= src * coeff over GF(2^16).
func galMulAdd(f *GF16, dst, src []byte, coeff uint16) {
	galMulAddScalar(f, dst, src, coeff)
}
