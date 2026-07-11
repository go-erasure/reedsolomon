//go:build ignore

// Command gen produces galois_arm64.s with go-asmgen: a NEON split-table region
// multiply over GF(2^16). It emits galMulNEON (dst = src*coeff) and
// galMulAddNEON (dst ^= src*coeff), each folding n whole 32-byte blocks (16
// big-endian uint16 words) per call.
//
// Method (GF-Complete SPLIT(16,4) with TBL byte-shuffle lookups). buildSplitTables
// (galois_tables.go) precomputes, for the fixed coeff, four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each, loaded into
// V8..V15. Per 32-byte block:
//
//   - VLD2 deinterleaves the interleaved big-endian words [hi0,lo0,hi1,lo1,...]
//     straight into a HI byte plane (V0) and a LO byte plane (V1) — NEON gives
//     the byte deinterleave for free, unlike amd64.
//   - Nibbles: n0=LO&0x0F, n1=LO>>4, n2=HI&0x0F, n3=HI>>4 (VUSHR fills zeros so
//     the high nibble needs no mask; VAND 0x0F for the low ones).
//   - Eight VTBL lookups into V8..V15 XORed (VEOR) into a LO-out and a HI-out
//     plane: LO-out=T0L[n0]^T1L[n1]^T2L[n2]^T3L[n3], HI-out the TkH analogue.
//   - VST2 re-interleaves HI-out,LO-out back to big-endian word order (galMulNEON),
//     or the existing dst is VLD2-read and XORed in first (galMulAddNEON).
//
// The kernel uses only long-available NEON (VLD2/VST2, VTBL, VUSHR, VAND, VEOR,
// VMOVI), so unlike the adler32 arm64 kernel it needs no go1.27 build tag. It
// runs natively on the arm64 dev host and CI, where the differential oracle
// proves it bit-exact.
//
// Signature: galMulNEON(dst, src []byte, tbl *[128]byte, n int), n = number of
// whole 32-byte blocks. Run: go run galois_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
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

// lookup accumulates Tk[nk] into acc via VTBL then VEOR (using V6 scratch),
// except the first term which seeds acc directly.
func lookup(b *arm64.Builder, table, index, acc string, first bool) {
	if first {
		b.Raw("VTBL %s, [%s], %s", index, table, acc)
		return
	}
	b.Raw("VTBL %s, [%s], V6.B16", index, table).
		Raw("VEOR V6.B16, %s, %s", acc, acc)
}

func gen(f *emit.File, name string, muladd bool) {
	b := arm64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").
		LoadArg("tbl", "R4").LoadArg("n", "R2").
		// Load the 8 coeff tables (128 bytes) into V8..V15 and the 0x0F mask.
		Raw("VLD1.P 64(R4), [V8.B16, V9.B16, V10.B16, V11.B16]").
		Raw("VLD1 (R4), [V12.B16, V13.B16, V14.B16, V15.B16]").
		Raw("VMOVI $15, V7.B16").
		Raw("CBZ R2, done").
		Label("loop").
		// Deinterleave 16 words into HI plane V0 and LO plane V1.
		Raw("VLD2.P 32(R1), [V0.B16, V1.B16]").
		// Nibbles n0=V2, n1=V3, n2=V4, n3=V5.
		Raw("VAND V7.B16, V1.B16, V2.B16").
		Raw("VUSHR $4, V1.B16, V3.B16").
		Raw("VAND V7.B16, V0.B16, V4.B16").
		Raw("VUSHR $4, V0.B16, V5.B16")
	// LO-out -> V19, HI-out -> V18 (kept consecutive for VST2 [hi=V18, lo=V19]).
	lookup(b, "V8.B16", "V2.B16", "V19.B16", true)   // T0L[n0]
	lookup(b, "V10.B16", "V3.B16", "V19.B16", false) // ^T1L[n1]
	lookup(b, "V12.B16", "V4.B16", "V19.B16", false) // ^T2L[n2]
	lookup(b, "V14.B16", "V5.B16", "V19.B16", false) // ^T3L[n3]
	lookup(b, "V9.B16", "V2.B16", "V18.B16", true)   // T0H[n0]
	lookup(b, "V11.B16", "V3.B16", "V18.B16", false) // ^T1H[n1]
	lookup(b, "V13.B16", "V4.B16", "V18.B16", false) // ^T2H[n2]
	lookup(b, "V15.B16", "V5.B16", "V18.B16", false) // ^T3H[n3]
	if muladd {
		b.
			Raw("VLD2 (R0), [V16.B16, V17.B16]").
			Raw("VEOR V16.B16, V18.B16, V18.B16").
			Raw("VEOR V17.B16, V19.B16, V19.B16")
	}
	b.
		Raw("VST2.P [V18.B16, V19.B16], 32(R0)").
		Raw("SUBS $1, R2, R2").Raw("BNE loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("arm64")
	gen(f, "galMulNEON", false)
	gen(f, "galMulAddNEON", true)
	if err := os.WriteFile("galois_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_arm64.s")
}
