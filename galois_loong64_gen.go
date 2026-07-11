//go:build ignore

// Command gen produces galois_loong64.s with go-asmgen: an LSX split-table region
// multiply over GF(2^16). It emits galMulLSX (dst = src*coeff) and galMulAddLSX
// (dst ^= src*coeff), each folding n whole 32-byte blocks (16 big-endian uint16
// words) per call.
//
// Method (GF-Complete SPLIT(16,4) with vshuf.b byte-shuffle lookups). buildSplitTables
// (galois_tables.go) precomputes, for the fixed coeff, four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each, loaded into
// V16..V23. Per 32-byte block:
//
//   - VMOVQ two 16-byte halves A,B of the interleaved big-endian words
//     [hi0,lo0,hi1,lo1,...]. LSX loads memory byte j into vector lane j (little-
//     endian element order does not renumber byte lanes), so lane j == byte j and
//     the byte-shuffle sees the shard in memory order.
//   - Deinterleave into a HI byte plane (even lanes) and a LO byte plane (odd
//     lanes) with two VSHUFB by the constant even/odd index vectors {0,2,..,30}
//     and {1,3,..,31} over the 32-lane A||B concatenation.
//   - Nibbles: n0=LO&0x0F, n1=LO>>4, n2=HI&0x0F, n3=HI>>4 (VSRLB shifts each byte
//     lane right, zero-filling, so the high nibble needs no mask; VANDB $15 masks
//     the low ones).
//   - Eight VSHUFB table lookups of the form `VSHUFB Tk,Tk,nk,out` (both source
//     operands the same 16-entry table register, index 0..15) XORed (VXORV) into a
//     LO-out and a HI-out plane.
//   - Re-interleave HI-out,LO-out back to big-endian word order with VILVLB/VILVHB
//     (interleave-low/high byte gives out bytes 0..15 / 16..31), then VMOVQ store
//     (galMulLSX) or XOR the existing dst in first (galMulAddLSX).
//
// LSX (128-bit) is the loong64 baseline (as in the sibling go-simd kernels), so no
// runtime feature check; the differential oracle proves it bit-exact under QEMU
// (QEMU_CPU=la464). No LASX is used.
//
// Signature: galMulLSX(dst, src []byte, tbl *[128]byte, n int), n = number of whole
// 32-byte blocks. Run: go run galois_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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

// gen emits one kernel. The loop advances R5 (src) and R4 (dst) by 32 and
// decrements R7 (block count).
func gen(f *emit.File, name string, muladd bool, even, odd string) {
	b := loong64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "R4").LoadArg("src_base", "R5").
		LoadArg("tbl", "R6").LoadArg("n", "R7").
		// Load the 8 coeff tables (128 bytes) into V16..V23.
		Raw("VMOVQ 0(R6), V16").Raw("VMOVQ 16(R6), V17").
		Raw("VMOVQ 32(R6), V18").Raw("VMOVQ 48(R6), V19").
		Raw("VMOVQ 64(R6), V20").Raw("VMOVQ 80(R6), V21").
		Raw("VMOVQ 96(R6), V22").Raw("VMOVQ 112(R6), V23").
		// Even/odd deinterleave selectors.
		Raw("MOVV $%s(SB), R8", even).Raw("VMOVQ (R8), V24").
		Raw("MOVV $%s(SB), R8", odd).Raw("VMOVQ (R8), V25").
		Raw("BEQ R7, R0, done").
		Label("loop").
		// A=V0 (lanes 0..15), B=V1 (lanes 16..31 of the A||B concatenation).
		Raw("VMOVQ 0(R5), V0").Raw("VMOVQ 16(R5), V1").
		// HI plane V2 = even lanes, LO plane V3 = odd lanes. vshuf.b Go operand
		// order is (va=index, vk=low-half, vj=high-half, vd): index first, then the
		// A||B concatenation (vk=A=lanes 0..15, vj=B=lanes 16..31).
		Raw("VSHUFB V24, V0, V1, V2").
		Raw("VSHUFB V25, V0, V1, V3").
		// Nibbles n0=V4, n1=V5, n2=V6, n3=V7.
		Raw("VANDB $15, V3, V4").
		Raw("VSRLB $4, V3, V5").
		Raw("VANDB $15, V2, V6").
		Raw("VSRLB $4, V2, V7").
		// LO-out plane V8 = T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3]. Lookup form: the nibble
		// is va (index), the 16-entry table is both vk and vj.
		Raw("VSHUFB V4, V16, V16, V8").
		Raw("VSHUFB V5, V18, V18, V9").Raw("VXORV V9, V8, V8").
		Raw("VSHUFB V6, V20, V20, V9").Raw("VXORV V9, V8, V8").
		Raw("VSHUFB V7, V22, V22, V9").Raw("VXORV V9, V8, V8").
		// HI-out plane V10 = T0H[n0]^T1H[n1]^T2H[n2]^T3H[n3].
		Raw("VSHUFB V4, V17, V17, V10").
		Raw("VSHUFB V5, V19, V19, V9").Raw("VXORV V9, V10, V10").
		Raw("VSHUFB V6, V21, V21, V9").Raw("VXORV V9, V10, V10").
		Raw("VSHUFB V7, V23, V23, V9").Raw("VXORV V9, V10, V10").
		// Re-interleave: out0 (V11)=ilvl(hi,lo)=bytes 0..15, out1 (V12)=ilvh.
		Raw("VILVLB V10, V8, V11").
		Raw("VILVHB V10, V8, V12")
	if muladd {
		b.
			Raw("VMOVQ 0(R4), V13").Raw("VXORV V13, V11, V11").
			Raw("VMOVQ 16(R4), V14").Raw("VXORV V14, V12, V12")
	}
	b.
		Raw("VMOVQ V11, 0(R4)").Raw("VMOVQ V12, 16(R4)").
		Raw("ADDV $32, R5").Raw("ADDV $32, R4").
		Raw("ADDV $-1, R7").Raw("BNE R7, R0, loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("loong64")
	even := f.Data("galEvenIdx", evenIdx())
	odd := f.Data("galOddIdx", oddIdx())
	gen(f, "galMulLSX", false, even, odd)
	gen(f, "galMulAddLSX", true, even, odd)
	if err := os.WriteFile("galois_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_loong64.s")
}
