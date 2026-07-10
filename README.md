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
- **100% test coverage**, verified across nine `GOOS/GOARCH` targets.

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
