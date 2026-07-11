//go:build amd64 || arm64

package reedsolomon

// simdBlock is the region granularity (in bytes) of the SIMD split-table
// kernels: 32 bytes == 16 big-endian uint16 words folded per inner iteration.
// The dispatch (galois_amd64.go / galois_arm64.go) feeds whole multiples of
// simdBlock to the assembly kernel and defers the remaining bytes to the scalar
// oracle (galMulScalar / galMulAddScalar).
const simdBlock = 32

// buildSplitTables fills tbl with the GF-Complete SPLIT(16,4) tables for a fixed
// coeff: for each nibble position k in 0..3, Tk[n] = (n<<4k) * coeff over
// GF(2^16), split into a low-byte table TkL and a high-byte table TkH of 16
// entries each. Then for any 16-bit word v with nibbles n0..n3,
//
//	v*coeff = T0[n0] ^ T1[n1] ^ T2[n2] ^ T3[n3]
//
// so the product's low byte is T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3] and its high byte
// the TkH analogue — each a PSHUFB/TBL byte-shuffle lookup in the kernels. The
// 128-byte layout is T0L,T0H,T1L,T1H,T2L,T2H,T3L,T3H (each 16 bytes); TkL lives
// at offset 32k and TkH at 32k+16, matching the offsets the generated assembly
// loads. It is rebuilt once per region call (256 field multiplies, cheap next to
// folding a whole shard).
func buildSplitTables(f *GF16, coeff uint16, tbl *[128]byte) {
	for k := 0; k < 4; k++ {
		shift := uint(4 * k)
		baseLo := 32 * k
		baseHi := 32*k + 16
		for nib := 0; nib < 16; nib++ {
			p := f.Mul(uint16(nib)<<shift, coeff)
			tbl[baseLo+nib] = byte(p)
			tbl[baseHi+nib] = byte(p >> 8)
		}
	}
}
