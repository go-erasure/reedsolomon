//go:build !amd64 && !arm64

package reedsolomon

// On architectures without a go-asmgen SIMD kernel (riscv64, ppc64le, s390x,
// loong64, wasm, 386, …) the region multiply routes entirely through the scalar
// oracle. These kernels are byte-for-byte identical to the SIMD fast paths by
// construction (the differential test proves it on amd64/arm64), so downstream
// behaviour is unchanged; only the throughput differs. A verified SIMD kernel
// for these targets is a real-hardware follow-up (cfarm POWER/RISC-V/loong,
// direct s390x) — see the package README.

// galMul writes dst = src * coeff over GF(2^16).
func galMul(f *GF16, dst, src []byte, coeff uint16) {
	galMulScalar(f, dst, src, coeff)
}

// galMulAdd accumulates dst ^= src * coeff over GF(2^16).
func galMulAdd(f *GF16, dst, src []byte, coeff uint16) {
	galMulAddScalar(f, dst, src, coeff)
}
