//go:build ignore

// Command gen produces galois_amd64.s with go-asmgen: SSSE3 split-table region
// multiply over GF(2^16). It emits galMulSSSE3 (dst = src*coeff) and
// galMulAddSSSE3 (dst ^= src*coeff), each folding n whole 32-byte blocks (16
// big-endian uint16 words) per call.
//
// Method (GF-Complete SPLIT(16,4) with PSHUFB byte-shuffle lookups). For a fixed
// coeff, buildSplitTables (galois_tables.go) precomputes four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each, so a word v
// with nibbles n0..n3 has v*coeff low byte = T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3] and
// high byte the TkH analogue. Per 32-byte block:
//
//   - MOVOU two 16-byte halves A,B of the interleaved big-endian words
//     [hi0,lo0,hi1,lo1,...].
//   - Deinterleave into a HI byte plane and a LO byte plane with one PSHUFB even
//     mask {0,2,4,6,8,10,12,14,0x80...} plus PUNPCKLQDQ; the LO plane is the same
//     shuffle after PSRLDQ $1.
//   - Extract nibbles: n0=LO&0x0F, n1=(LO>>4)&0x0F, n2=HI&0x0F, n3=(HI>>4)&0x0F
//     (PSRLW $4 shifts 16-bit lanes; PAND 0x0F masks the borrowed bits).
//   - Eight PSHUFB lookups into the coeff tables (loaded from *tbl via MOVOU, so
//     no 16-byte alignment is required) XORed into a LO-out and a HI-out plane.
//   - Re-interleave with PUNPCKLBW/PUNPCKHBW back to big-endian word order and
//     MOVOU-store (galMulSSSE3) or XOR into the existing dst first (galMulAddSSSE3).
//
// PSHUFB is the SSSE3 form of the same byte-shuffle a 256-bit AVX2 VPSHUFB kernel
// would use; the 128-bit SSSE3 encoding is chosen because it validates natively
// on the amd64 CI runner AND under Rosetta on the arm64 dev host (which has no
// AVX2), so the differential oracle proves it end-to-end everywhere. Widening to
// AVX2/VPSHUFB over 64-byte blocks (with cross-lane deinterleave) is a follow-up.
//
// Signature: galMulSSSE3(dst, src []byte, tbl *[128]byte, n int), n = number of
// whole 32-byte blocks. Run: go run galois_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Slice("dst"), abi.Slice("src"),
			abi.Scalar("tbl", abi.Ptr), abi.Scalar("n", abi.Int64),
		},
		nil,
	)
}

func rep8(b byte) []byte {
	out := make([]byte, 16)
	for i := range out {
		out[i] = b
	}
	return out
}

// evenMask gathers the even bytes of a 16-byte vector into its low 8 lanes and
// zeroes the high 8 (0x80 clears a lane in PSHUFB).
func evenMask() []byte {
	m := make([]byte, 16)
	for i := 0; i < 8; i++ {
		m[i] = byte(2 * i)
	}
	for i := 8; i < 16; i++ {
		m[i] = 0x80
	}
	return m
}

// body emits the shared per-block compute and the store phase (XOR-into-dst when
// muladd). src halves come from SI, dst from DX, tables from DI.
func body(b *amd64.Builder, muladd bool) {
	b.
		// A = words' first 16 bytes, B = next 16 bytes (both interleaved hi,lo).
		Raw("MOVOU 0(SI), X0").Raw("MOVOU 16(SI), X1").
		// HI plane (X2) = even bytes of A||B via one shuffle mask + PUNPCKLQDQ.
		Raw("MOVO X0, X2").Raw("PSHUFB X6, X2").
		Raw("MOVO X1, X3").Raw("PSHUFB X6, X3").
		Raw("PUNPCKLQDQ X3, X2").
		// LO plane (X0) = odd bytes = even bytes after a 1-byte down-shift.
		Raw("PSRLDQ $1, X0").Raw("PSHUFB X6, X0").
		Raw("PSRLDQ $1, X1").Raw("PSHUFB X6, X1").
		Raw("PUNPCKLQDQ X1, X0").
		// Nibbles: n0=X3, n1=X0, n2=X4, n3=X2.
		Raw("MOVO X0, X3").Raw("PAND X7, X3").
		Raw("PSRLW $4, X0").Raw("PAND X7, X0").
		Raw("MOVO X2, X4").Raw("PAND X7, X4").
		Raw("PSRLW $4, X2").Raw("PAND X7, X2").
		// LO-out plane (X5) = T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3].
		Raw("MOVOU 0(DI), X5").Raw("PSHUFB X3, X5").
		Raw("MOVOU 32(DI), X1").Raw("PSHUFB X0, X1").Raw("PXOR X1, X5").
		Raw("MOVOU 64(DI), X1").Raw("PSHUFB X4, X1").Raw("PXOR X1, X5").
		Raw("MOVOU 96(DI), X1").Raw("PSHUFB X2, X1").Raw("PXOR X1, X5").
		// HI-out plane (X8) = T0H[n0]^T1H[n1]^T2H[n2]^T3H[n3].
		Raw("MOVOU 16(DI), X8").Raw("PSHUFB X3, X8").
		Raw("MOVOU 48(DI), X1").Raw("PSHUFB X0, X1").Raw("PXOR X1, X8").
		Raw("MOVOU 80(DI), X1").Raw("PSHUFB X4, X1").Raw("PXOR X1, X8").
		Raw("MOVOU 112(DI), X1").Raw("PSHUFB X2, X1").Raw("PXOR X1, X8").
		// Re-interleave: words 0..7 = PUNPCKLBW(HI,LO), words 8..15 = PUNPCKHBW.
		Raw("MOVO X8, X9").
		Raw("PUNPCKLBW X5, X8").
		Raw("PUNPCKHBW X5, X9")
	if muladd {
		b.
			Raw("MOVOU 0(DX), X1").Raw("PXOR X1, X8").
			Raw("MOVOU 16(DX), X1").Raw("PXOR X1, X9")
	}
	b.
		Raw("MOVOU X8, 0(DX)").Raw("MOVOU X9, 16(DX)")
}

func gen(f *emit.File, name string, muladd bool, even, lo string) {
	b := amd64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "DX").LoadArg("src_base", "SI").
		LoadArg("tbl", "DI").LoadArg("n", "CX").
		Raw("MOVOU %s+0(SB), X6", even).
		Raw("MOVOU %s+0(SB), X7", lo).
		Raw("TESTQ CX, CX").Raw("JZ done").
		Label("loop")
	body(b, muladd)
	b.
		Raw("ADDQ $32, SI").Raw("ADDQ $32, DX").
		Raw("SUBQ $1, CX").Raw("JNZ loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("amd64")
	even := f.Data("evenMask", evenMask())
	lo := f.Data("loNib", rep8(0x0f))
	gen(f, "galMulSSSE3", false, even, lo)
	gen(f, "galMulAddSSSE3", true, even, lo)
	if err := os.WriteFile("galois_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_amd64.s")
}
