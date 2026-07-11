//go:build ignore

// Command gen produces galois_s390x.s with go-asmgen: a z13 vector-facility
// split-table region multiply over GF(2^16). It emits galMulVX (dst = src*coeff)
// and galMulAddVX (dst ^= src*coeff), each folding n whole 32-byte blocks (16
// big-endian uint16 words) per call.
//
// Method (GF-Complete SPLIT(16,4) with VPERM byte-shuffle lookups). buildSplitTables
// (galois_tables.go) precomputes, for the fixed coeff, four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each, loaded into
// V16..V23. Per 32-byte block:
//
//   - VL two 16-byte halves A,B of the interleaved big-endian words
//     [hi0,lo0,hi1,lo1,...]. s390x is big-endian, so VL places memory byte j at
//     the vector's big-endian byte position j — exactly the numbering VPERM uses.
//   - Deinterleave into a HI byte plane and a LO byte plane with two VPERM by the
//     constant even/odd index vectors {0,2,..,30} and {1,3,..,31} (each selecting
//     from the 32-byte A||B concatenation).
//   - Nibbles: n0=LO&0x0F, n1=LO>>4, n2=HI&0x0F, n3=HI>>4 (VESRLB shifts each byte
//     element right, zero-filling, so the high nibble needs no mask; VN with a
//     VREPIB-splatted 0x0F masks the low ones).
//   - Eight VPERM table lookups (index 0..15 selects the low half of the table
//     register, which holds the 16-entry Tk) XORed (VX) into a LO-out and a HI-out
//     plane: LO-out=T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3], HI-out the TkH analogue.
//   - Re-interleave HI-out,LO-out back to big-endian word order with VMRHB/VMRLB
//     (merge-high/low byte gives out bytes 0..15 / 16..31), then VST (galMulVX) or
//     XOR the existing dst in first (galMulAddVX).
//
// The vector facility is the s390x (z13) baseline, so no runtime feature check and
// no SIGILL. Only base-facility ops are used (VL/VST/VPERM/VN/VX/VESRLB/VREPIB/
// VMRHB/VMRLB), which QEMU's default s390x model implements; the differential
// oracle proves it bit-exact under emulation.
//
// Signature: galMulVX(dst, src []byte, tbl *[128]byte, n int), n = number of whole
// 32-byte blocks. Run: go run galois_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
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

// evenIdx / oddIdx select the even (HI) / odd (LO) bytes of the 32-byte A||B
// concatenation in a VPERM.
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

// gen emits one kernel. The loop advances R2 (src) and R1 (dst) by 32 and
// decrements R4 (block count).
func gen(f *emit.File, name string, muladd bool, even, odd string) {
	b := s390x.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").
		LoadArg("tbl", "R3").LoadArg("n", "R4").
		// Load the 8 coeff tables (128 bytes) into V16..V23.
		Raw("VL 0(R3), V16").Raw("VL 16(R3), V17").
		Raw("VL 32(R3), V18").Raw("VL 48(R3), V19").
		Raw("VL 64(R3), V20").Raw("VL 80(R3), V21").
		Raw("VL 96(R3), V22").Raw("VL 112(R3), V23").
		// 0x0F nibble mask (V24) and the even/odd deinterleave selectors.
		Raw("VREPIB $0x0f, V24").
		Raw("MOVD $%s(SB), R5", even).Raw("VL 0(R5), V25").
		Raw("MOVD $%s(SB), R5", odd).Raw("VL 0(R5), V26").
		Raw("CMPBEQ R4, $0, done").
		Label("loop").
		// A=V0 (bytes 0..15), B=V1 (bytes 16..31).
		Raw("VL 0(R2), V0").Raw("VL 16(R2), V1").
		// HI plane V2 = even bytes, LO plane V3 = odd bytes.
		Raw("VPERM V0, V1, V25, V2").
		Raw("VPERM V0, V1, V26, V3").
		// Nibbles n0=V4, n1=V5, n2=V6, n3=V7.
		Raw("VN V24, V3, V4").
		Raw("VESRLB $4, V3, V5").
		Raw("VN V24, V2, V6").
		Raw("VESRLB $4, V2, V7").
		// LO-out plane V8 = T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3].
		Raw("VPERM V16, V16, V4, V8").
		Raw("VPERM V18, V18, V5, V9").Raw("VX V9, V8, V8").
		Raw("VPERM V20, V20, V6, V9").Raw("VX V9, V8, V8").
		Raw("VPERM V22, V22, V7, V9").Raw("VX V9, V8, V8").
		// HI-out plane V10 = T0H[n0]^T1H[n1]^T2H[n2]^T3H[n3].
		Raw("VPERM V17, V17, V4, V10").
		Raw("VPERM V19, V19, V5, V9").Raw("VX V9, V10, V10").
		Raw("VPERM V21, V21, V6, V9").Raw("VX V9, V10, V10").
		Raw("VPERM V23, V23, V7, V9").Raw("VX V9, V10, V10").
		// Re-interleave: out0 (V11)=merge-high(HI,LO)=bytes 0..15, out1 (V12)=low.
		Raw("VMRHB V10, V8, V11").
		Raw("VMRLB V10, V8, V12")
	if muladd {
		b.
			Raw("VL 0(R1), V13").Raw("VX V13, V11, V11").
			Raw("VL 16(R1), V14").Raw("VX V14, V12, V12")
	}
	b.
		Raw("VST V11, 0(R1)").Raw("VST V12, 16(R1)").
		Raw("ADD $32, R2").Raw("ADD $32, R1").
		Raw("ADD $-1, R4").Raw("CMPBNE R4, $0, loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("s390x")
	even := f.Data("galEvenIdx", evenIdx())
	odd := f.Data("galOddIdx", oddIdx())
	gen(f, "galMulVX", false, even, odd)
	gen(f, "galMulAddVX", true, even, odd)
	if err := os.WriteFile("galois_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_s390x.s")
}
