<p align="center"><img src="https://raw.githubusercontent.com/go-erasure/brand/main/social/go-erasure.png" alt="go-erasure/reedsolomon" width="720"></p>

# reedsolomon

[![CI](https://github.com/go-erasure/reedsolomon/actions/workflows/ci.yml/badge.svg)](https://github.com/go-erasure/reedsolomon/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-erasure/reedsolomon.svg)](https://pkg.go.dev/github.com/go-erasure/reedsolomon)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Pure-Go, dependency-free **Reed-Solomon erasure code over GF(2¹⁶)**. It works
both as a general erasure codec for storage redundancy and as the arithmetic
core for [PAR2](https://en.wikipedia.org/wiki/Parchive).

- **CGO_ENABLED=0**, standard library only, no third-party dependencies.
- **MDS** systematic code: a Cauchy generator matrix guarantees any `dataShards`
  of the `dataShards + parityShards` shards reconstruct the original data.
- **PAR2-compatible field**: GF(2¹⁶) with primitive polynomial `0x1100B` and
  generator `2`. The exported [`GF16`](gf16.go) type can be reused by a PAR2
  layer built on top of this package.
- **SIMD region multiply**: the field-region hot loops (`galMul`/`galMulAdd`,
  which is all `Encode`/`Verify`/`Reconstruct` spend their time in) have a
  go-asmgen split-table fast path on all six 64-bit targets — amd64, arm64,
  s390x, ppc64le, riscv64 and loong64 — each proven bit-identical to the scalar
  loop by a differential oracle (under QEMU for the emulated arches). See
  [Performance](#performance).
- **100% test coverage**, verified across nine `GOOS/GOARCH` targets.

## Performance

The region multiply uses the GF-Complete **SPLIT(16,4)** technique: for a fixed
coefficient, four low-byte and four high-byte 16-entry nibble tables turn a
GF(2¹⁶) product into eight byte-shuffle lookups XORed together, so 16 words are
multiplied per iteration instead of one at a time. Each arch expresses the same
math with its native byte-permute: deinterleave the big-endian words into HI/LO
byte planes, extract the four nibbles, do the eight table lookups, XOR, and
re-interleave. The Go dispatch builds the per-coefficient tables (256 field
multiplies, cheap next to folding a whole shard) and hands them to the assembly
kernel; the sub-block tail falls back to the scalar loop.

| Arch | Kernel | Verification |
| --- | --- | --- |
| **amd64** | SSSE3 `PSHUFB`, 128-bit, 32 B/iter | native CI + Rosetta |
| **arm64** | NEON `VLD2`/`TBL`/`VST2`, 32 B/iter | native CI + dev host |
| **s390x** | z13 vector `VPERM` + `VMRHB`/`VMRLB` | real z15 (differential + 100% cov) |
| **ppc64le** | POWER8 VSX `LXVD2X`/`VPERM`/`STXVD2X` | real POWER8E (no SIGILL) + power8 ISA guard |
| **riscv64** | RVV `vlseg2e8`/`vrgather.vv`, VLEN-agnostic | real X60 RVV1.0 VLEN=256 |
| **loong64** | LSX `vshuf.b` + `vilvl.b`/`vilvh.b` | real Loongson LSX |

The ppc64le kernel is strictly POWER8-baseline (no ISA-3.0 `LXVB16X`); a dedicated
CI lane runs it under `QEMU_CPU=power8` to prove it never `SIGILL`s. The riscv64
kernel dispatches only when the V extension is present (`cpu.RISCV64.HasV`) and is
byte-granular (segment loads), so it is VLEN-agnostic and free of the misaligned
wider-load trap.

### Measured on real hardware (`galMulAdd`, 1 MiB region)

Every kernel is proven byte-identical to the scalar oracle **and** benchmarked on
real silicon — not QEMU (whose perf is emulation-serialized and meaningless):

| Arch    | Hardware                      | Scalar | SIMD | Speedup |
|---------|-------------------------------|--------|------|---------|
| arm64   | Apple M4 Max (NEON)           | 1.45 GB/s | 19.5 GB/s | ~13.4× |
| ppc64le | POWER8E (VSX `VPERM`)         | 202 MB/s  | 3.41 GB/s | **16.9×** |
| loong64 | Loongson 3C5000L (LSX)        | 255 MB/s  | 4.26 GB/s | **16.7×** |
| s390x   | IBM z15 (vector `VPERM`)      | 679 MB/s  | 9.49 GB/s | **14.0×** |
| riscv64 | SpacemiT X60, RVV1.0 VLEN=256 | 18 MB/s   | 1.11 GB/s | **61.1×** |

amd64 (SSSE3) is verified native on CI. The riscv64 landslide reflects how slow a
scalar GF(2¹⁶) multiply is on that in-order core — exactly where the table-lookup
SIMD pays off most.

## Install

```sh
go get github.com/go-erasure/reedsolomon
```

Requires Go 1.26.4 or newer.

## Example

```go
package main

import (
	"bytes"
	"fmt"

	"github.com/go-erasure/reedsolomon"
)

func main() {
	// 4 data shards + 2 parity shards: any 4 of the 6 shards recover everything.
	enc, err := reedsolomon.New(4, 2)
	if err != nil {
		panic(err)
	}

	// Shards are byte slices read as big-endian uint16 words; all shards must
	// share the same even length. The parity shards are written by Encode.
	shards := make([][]byte, 6)
	for i := range shards {
		shards[i] = make([]byte, 8)
	}
	copy(shards[0], []byte("data0..."))
	copy(shards[1], []byte("data1..."))
	copy(shards[2], []byte("data2..."))
	copy(shards[3], []byte("data3..."))

	if err := enc.Encode(shards); err != nil {
		panic(err)
	}

	// Simulate the loss of two shards (one data, one parity).
	original := append([]byte(nil), shards[1]...)
	present := []bool{true, false, true, true, false, true}
	for i, ok := range present {
		if !ok {
			for k := range shards[i] {
				shards[i][k] = 0 // erased slice must stay allocated to shard length
			}
		}
	}

	if err := enc.Reconstruct(shards, present); err != nil {
		panic(err)
	}
	fmt.Println("recovered:", bytes.Equal(shards[1], original)) // true
}
```

## API

```go
func New(dataShards, parityShards int) (*Encoder, error)

func (e *Encoder) Encode(shards [][]byte) error
func (e *Encoder) Verify(shards [][]byte) (bool, error)
func (e *Encoder) Reconstruct(shards [][]byte, present []bool) error
```

Shards are interpreted as big-endian `uint16` words, so every shard must have
the same **even** length. `Reconstruct` succeeds whenever at least `dataShards`
shards are present; each erased shard's slice must remain allocated to the
common shard length.

### The finite field

`GF16` implements GF(2¹⁶) with primitive polynomial `0x1100B` and generator `2`,
the exact field used by PAR2:

```go
f := reedsolomon.NewGF16()
f.Add(a, b) // XOR
f.Mul(a, b)
f.Div(a, b) // panics if b == 0
f.Exp(power)
f.Pow(a, n)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE). Copyright the go-erasure/reedsolomon
authors.
