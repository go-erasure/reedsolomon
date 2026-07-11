//go:build ignore

// Command gen produces galois_riscv64.s with go-asmgen: an RVV (vector) split-table
// region multiply over GF(2^16). It emits galMulRVV (dst = src*coeff) and
// galMulAddRVV (dst ^= src*coeff), each folding n whole 32-byte blocks (16
// big-endian uint16 words) per call.
//
// Method (GF-Complete SPLIT(16,4) with vrgather.vv table lookups). buildSplitTables
// (galois_tables.go) precomputes, for the fixed coeff, four low-byte tables
// T0L..T3L and four high-byte tables T0H..T3H of 16 entries each, loaded (VLE8V,
// vl=16) into V16..V23. The block count is turned into a word count (n<<4) and the
// region is walked in VSETVLI strips (SEW=8, LMUL=1); at VLEN=256 a strip is up to
// 32 words. Per strip:
//
//   - VLSEG2E8V byte-segment-loads the interleaved big-endian words
//     [hi0,lo0,hi1,lo1,...] straight into a HI byte plane (V0=field0=even bytes)
//     and a LO byte plane (V1=field1=odd bytes) — the RVV analogue of NEON VLD2,
//     and byte-granular so it never triggers the misaligned wider-load SIGBUS.
//   - Nibbles: n0=LO&0x0F, n1=LO>>4, n2=HI&0x0F, n3=HI>>4 (VSRLVI zero-fills, so the
//     high nibble needs no mask; VANDVI $15 masks the low ones).
//   - Eight VRGATHERVV lookups vd[i]=Tk[nk[i]] (index vector is a nibble plane 0..15,
//     source is the 16-entry table register; the table registers' elements >=16 are
//     never indexed so their tail is don't-care) XORed (VXORVV) into a LO-out and a
//     HI-out plane.
//   - VSSEG2E8V re-interleaves HI-out(field0),LO-out(field1) back to big-endian word
//     order (galMulRVV), or the existing dst is VLSEG2E8V-read and XORed in first
//     (galMulAddRVV).
//
// The V (vector) instructions trap on a hart without the extension, so the Go
// wrapper (galois_riscv64.go) only dispatches here when cpu.RISCV64.HasV is set,
// else the whole region goes through the scalar oracle. RVV assembly requires the
// go1.26 toolchain, which the module's go 1.26.4 floor guarantees. The differential
// oracle proves it bit-exact under QEMU (rv64,v=true,vlen=256).
//
// Signature: galMulRVV(dst, src []byte, tbl *[128]byte, n int), n = number of whole
// 32-byte blocks. Run: go run galois_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
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

// gen emits one kernel. X5=dst, X6=src, X7=tbl, X8=n(blocks), X9=words remaining,
// X10=scratch, X11=granted vl, X12=byte stride.
func gen(f *emit.File, name string, muladd bool) {
	b := riscv64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "X5").LoadArg("src_base", "X6").
		LoadArg("tbl", "X7").LoadArg("n", "X8").
		// words = blocks * 16.
		Raw("SLLI $4, X8, X9").
		// Load the 8 coeff tables (16 bytes each) into V16..V23 with vl=16.
		Raw("MOV $16, X10").
		Raw("VSETVLI X10, E8, M1, TA, MA, X10").
		Raw("VLE8V (X7), V16").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V17").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V18").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V19").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V20").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V21").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V22").Raw("ADD $16, X7, X7").
		Raw("VLE8V (X7), V23").
		Label("loop").
		Raw("BEQZ X9, done").
		// Grant a strip of up to VLMAX words; X11 = vl (words this strip).
		Raw("VSETVLI X9, E8, M1, TA, MA, X11").
		// Deinterleave: V0=HI plane (field0/even bytes), V1=LO plane (field1/odd).
		Raw("VLSEG2E8V (X6), V0").
		// Nibbles n0=V2, n1=V3, n2=V4, n3=V5.
		Raw("VANDVI $15, V1, V2").
		Raw("VSRLVI $4, V1, V3").
		Raw("VANDVI $15, V0, V4").
		Raw("VSRLVI $4, V0, V5").
		// LO-out plane V9 (kept adjacent to HI-out V8 for the segment store).
		Raw("VRGATHERVV V2, V16, V9").
		Raw("VRGATHERVV V3, V18, V6").Raw("VXORVV V6, V9, V9").
		Raw("VRGATHERVV V4, V20, V6").Raw("VXORVV V6, V9, V9").
		Raw("VRGATHERVV V5, V22, V6").Raw("VXORVV V6, V9, V9").
		// HI-out plane V8.
		Raw("VRGATHERVV V2, V17, V8").
		Raw("VRGATHERVV V3, V19, V6").Raw("VXORVV V6, V8, V8").
		Raw("VRGATHERVV V4, V21, V6").Raw("VXORVV V6, V8, V8").
		Raw("VRGATHERVV V5, V23, V6").Raw("VXORVV V6, V8, V8")
	if muladd {
		b.
			// Read existing dst (V10=HI, V11dst overlaps? use V10,V11) and XOR in.
			Raw("VLSEG2E8V (X5), V10").
			Raw("VXORVV V10, V8, V8").
			Raw("VXORVV V11, V9, V9")
	}
	b.
		// Re-interleave HI-out(field0),LO-out(field1) and store.
		Raw("VSSEG2E8V V8, (X5)").
		// stride = vl words * 2 bytes.
		Raw("SLLI $1, X11, X12").
		Raw("ADD X12, X6, X6").
		Raw("ADD X12, X5, X5").
		Raw("SUB X11, X9, X9").
		Raw("JMP loop").
		Label("done").Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("riscv64")
	gen(f, "galMulRVV", false)
	gen(f, "galMulAddRVV", true)
	if err := os.WriteFile("galois_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote galois_riscv64.s")
}
