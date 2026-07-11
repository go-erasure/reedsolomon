//go:build ignore

// Command gen produces galois_ppc64le.s with go-asmgen: a POWER8-baseline VSX/VMX
// split-table region multiply over GF(2^16). It emits galMulVSX (dst = src*coeff)
// and galMulAddVSX (dst ^= src*coeff), each folding n whole 32-byte blocks (16
// big-endian uint16 words) per call.
//
// Method (GF-Complete SPLIT(16,4) with VPERM byte-shuffle lookups). buildSplitTables
// (galois_tables.go) precomputes, for the fixed coeff, four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each. Everything is
// loaded with the POWER8 LXVD2X (no ISA-3.0 LXVB16X) and stored with STXVD2X.
//
// LE byte-order handling. On ppc64le LXVD2X places memory byte j at VMX big-endian
// byte position P(j) where P reverses the bytes within each doubleword
// (P(i)=7-i for i<8, 23-i for i>=8) — measured directly with MFVSRD. Loading the
// identity vector {0,1,...,15} through LXVD2X therefore yields exactly that
// permutation, giving a byte-reverse selector "bswap" for free: VPERM(x,x,bswap)
// rewrites any LXVD2X-loaded register into memory (big-endian) byte order. After
// correcting the inputs (and the tables and the deinterleave/interleave selectors,
// all correct-loaded the same way), the kernel is byte-for-byte the same as the
// s390x/arm64 ones. Per 32-byte block:
//
//   - LXVD2X two 16-byte halves A,B; VPERM-correct them to memory order.
//   - Deinterleave into a HI byte plane and a LO byte plane with two VPERM by the
//     even/odd selectors {0,2,..,30}/{1,3,..,31} over the A||B concatenation.
//   - Nibbles: n0=LO&0x0F, n1=LO>>4, n2=HI&0x0F, n3=HI>>4 (VSRB by a VSPLTISB-4
//     splat, VAND with a VSPLTISB-0x0F splat).
//   - Eight VPERM lookups `VPERM Tk,Tk,nk,out` (index 0..15 selects the 16-entry
//     table) XORed (VXOR) into a LO-out and a HI-out plane.
//   - Re-interleave HI-out,LO-out with two VPERM by interleave selectors, VPERM-
//     uncorrect back to LXVD2X order, and STXVD2X (galMulVSX) or XOR the existing
//     dst — itself LXVD2X-loaded and corrected — in first (galMulAddVSX).
//
// Only POWER8 (ISA 2.07) VSX/VMX ops are used (LXVD2X/STXVD2X/VPERM/VAND/VSRB/
// VXOR/VSPLTISB), so the kernel must not SIGILL on a POWER8 model; the CI power8
// lane enforces that and the differential oracle proves it bit-exact under QEMU.
//
// Signature: galMulVSX(dst, src []byte, tbl *[128]byte, n int), n = number of whole
// 32-byte blocks. Run: go run galois_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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

func iota16() []byte {
	m := make([]byte, 16)
	for i := range m {
		m[i] = byte(i)
	}
	return m
}

func evenIdx() []byte {
	m := make([]byte, 16)
	for i := 0; i < 16; i++ {
		m[i] = byte(2 * i)
	}
	return m
}

func oddIdx() []byte {
	m := make([]byte, 16)
	for i := 0; i < 16; i++ {
		m[i] = byte(2*i + 1)
	}
	return m
}

// il0 / il1 re-interleave HI-out,LO-out into bytes 0..15 / 16..31.
func ilIdx(hi int) []byte {
	m := make([]byte, 16)
	for k := 0; k < 8; k++ {
		m[2*k] = byte(hi*8 + k)        // HI-out[hi*8+k]  (VRA)
		m[2*k+1] = byte(16 + hi*8 + k) // LO-out[hi*8+k]  (VRB)
	}
	return m
}

// gen emits one kernel. R3=dst, R4=src, R5=tbl, R6=n(blocks); R7=const addr,
// R8=16 (second-half index), R0=0.
func gen(f *emit.File, name string, muladd bool, ident, even, odd, il0, il1 string) {
	b := ppc64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").
		LoadArg("tbl", "R5").LoadArg("n", "R6").
		Raw("MOVD $16, R8").
		// bswap corrector (V30) = LXVD2X of {0..15}; corrects any load to mem order.
		Raw("MOVD $%s(SB), R7", ident).Raw("LXVD2X (R7)(R0), VS62").
		// nibble mask (V28) and shift-by-4 (V29) splats.
		Raw("VSPLTISB $15, V28").
		Raw("VSPLTISB $4, V29").
		// Correct-load the deinterleave/interleave selectors (V24..V27).
		Raw("MOVD $%s(SB), R7", even).Raw("LXVD2X (R7)(R0), VS56").Raw("VPERM V24, V24, V30, V24").
		Raw("MOVD $%s(SB), R7", odd).Raw("LXVD2X (R7)(R0), VS57").Raw("VPERM V25, V25, V30, V25").
		Raw("MOVD $%s(SB), R7", il0).Raw("LXVD2X (R7)(R0), VS58").Raw("VPERM V26, V26, V30, V26").
		Raw("MOVD $%s(SB), R7", il1).Raw("LXVD2X (R7)(R0), VS59").Raw("VPERM V27, V27, V30, V27").
		// Correct-load the 8 coeff tables into V16..V23 (VS48..VS55).
		Raw("LXVD2X (R5)(R0), VS48").Raw("VPERM V16, V16, V30, V16").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS49").Raw("VPERM V17, V17, V30, V17").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS50").Raw("VPERM V18, V18, V30, V18").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS51").Raw("VPERM V19, V19, V30, V19").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS52").Raw("VPERM V20, V20, V30, V20").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS53").Raw("VPERM V21, V21, V30, V21").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS54").Raw("VPERM V22, V22, V30, V22").Raw("ADD $16, R5").
		Raw("LXVD2X (R5)(R0), VS55").Raw("VPERM V23, V23, V30, V23").
		Raw("CMP R6, $0").Raw("BEQ done").
		Label("loop").
		// Load and correct A (V0), B (V1) to memory order.
		Raw("LXVD2X (R4)(R0), VS32").Raw("VPERM V0, V0, V30, V0").
		Raw("LXVD2X (R4)(R8), VS33").Raw("VPERM V1, V1, V30, V1").
		// HI plane V2 = even bytes, LO plane V3 = odd bytes of A||B.
		Raw("VPERM V0, V1, V24, V2").
		Raw("VPERM V0, V1, V25, V3").
		// Nibbles n0=V4, n1=V5, n2=V6, n3=V7.
		Raw("VAND V3, V28, V4").
		Raw("VSRB V3, V29, V5").
		Raw("VAND V2, V28, V6").
		Raw("VSRB V2, V29, V7").
		// LO-out plane V8 = T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3].
		Raw("VPERM V16, V16, V4, V8").
		Raw("VPERM V18, V18, V5, V14").Raw("VXOR V14, V8, V8").
		Raw("VPERM V20, V20, V6, V14").Raw("VXOR V14, V8, V8").
		Raw("VPERM V22, V22, V7, V14").Raw("VXOR V14, V8, V8").
		// HI-out plane V9 = T0H[n0]^T1H[n1]^T2H[n2]^T3H[n3].
		Raw("VPERM V17, V17, V4, V9").
		Raw("VPERM V19, V19, V5, V14").Raw("VXOR V14, V9, V9").
		Raw("VPERM V21, V21, V6, V14").Raw("VXOR V14, V9, V9").
		Raw("VPERM V23, V23, V7, V14").Raw("VXOR V14, V9, V9").
		// Re-interleave: out0 (V10)=bytes 0..15, out1 (V11)=bytes 16..31 (mem order).
		Raw("VPERM V9, V8, V26, V10").
		Raw("VPERM V9, V8, V27, V11")
	if muladd {
		b.
			// XOR the existing dst (LXVD2X-loaded, corrected to mem order) in.
			Raw("LXVD2X (R3)(R0), VS44").Raw("VPERM V12, V12, V30, V12").
			Raw("LXVD2X (R3)(R8), VS45").Raw("VPERM V13, V13, V30, V13").
			Raw("VXOR V10, V12, V10").
			Raw("VXOR V11, V13, V11")
	}
	b.
		// Un-correct to LXVD2X order and store.
		Raw("VPERM V10, V10, V30, V10").Raw("STXVD2X VS42, (R3)(R0)").
		Raw("VPERM V11, V11, V30, V11").Raw("STXVD2X VS43, (R3)(R8)").
		Raw("ADD $32, R4").Raw("ADD $32, R3").
		Raw("ADD $-1, R6").Raw("CMP R6, $0").Raw("BNE loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("ppc64le")
	ident := f.Data("galIota16", iota16())
	even := f.Data("galEvenIdx", evenIdx())
	odd := f.Data("galOddIdx", oddIdx())
	il0 := f.Data("galIl0", ilIdx(0))
	il1 := f.Data("galIl1", ilIdx(1))
	gen(f, "galMulVSX", false, ident, even, odd, il0, il1)
	gen(f, "galMulAddVSX", true, ident, even, odd, il0, il1)
	if err := os.WriteFile("galois_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_ppc64le.s")
}
