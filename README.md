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
  go-asmgen split-table fast path on amd64 and arm64 — order-of-magnitude faster
  than the scalar loop, and proven bit-identical to it. See
  [Performance](#performance).
- **100% test coverage**, verified across nine `GOOS/GOARCH` targets.

## Performance

The region multiply uses the GF-Complete **SPLIT(16,4)** technique: for a fixed
coefficient, four low-byte and four high-byte 16-entry nibble tables turn a
GF(2¹⁶) product into eight `PSHUFB`/`TBL` byte-shuffle lookups XORed together, so
16 words are multiplied per vector instruction instead of one at a time. The Go
dispatch builds the per-coefficient tables (256 field multiplies, cheap next to
folding a whole shard) and hands them to the assembly kernel; the sub-block tail
falls back to the scalar loop.

| Arch | Kernel | Status |
| --- | --- | --- |
| **amd64** | SSSE3 `PSHUFB`, 128-bit, 32 B/iter | verified (native CI + Rosetta) |
| **arm64** | NEON `VLD2`/`TBL`/`VST2`, 32 B/iter | verified (native CI + dev host) |
| riscv64, ppc64le, s390x, loong64 | scalar fallback | real-hardware follow-up |

`galMulAdd` over a 1 MiB region, dev host (Apple arm64):

```
BenchmarkGalMulAddScalar   1431 MB/s
BenchmarkGalMulAddSIMD    19379 MB/s   (~13.5x, NEON)
```

The other four 64-bit targets build and pass on the scalar fallback today; a
verified SIMD kernel for them (RVV / VSX / vector-facility / LSX on cfarm and
direct s390x hardware) is a follow-up — an unverified kernel is worse than
scalar, so none is shipped until its differential oracle passes on real silicon.
The scalar loop and the SIMD kernels are checked byte-for-byte against each other
by a differential fuzz oracle on every CI run.

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
