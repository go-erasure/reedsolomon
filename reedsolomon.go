package reedsolomon

import (
	"bytes"
	"errors"
)

// Errors returned by the package.
var (
	// ErrShardCount is returned when the number of shards (or present flags)
	// does not equal dataShards+parityShards.
	ErrShardCount = errors.New("reedsolomon: wrong number of shards")
	// ErrShardSize is returned when shards are not all equal length or their
	// length is odd (shards are big-endian uint16 words).
	ErrShardSize = errors.New("reedsolomon: shards must be equal, even-length")
	// ErrTooFewShards is returned when fewer than dataShards shards are present
	// during reconstruction.
	ErrTooFewShards = errors.New("reedsolomon: not enough shards present to reconstruct")
	// ErrInvalidParams is returned by New for non-positive shard counts or when
	// dataShards+parityShards exceeds 65535.
	ErrInvalidParams = errors.New("reedsolomon: invalid data/parity shard counts")
)

// Encoder is an (n = dataShards + parityShards) Reed-Solomon erasure coder over
// GF(2^16). Shards are byte slices interpreted as big-endian uint16 words; all
// shards must share the same even length.
type Encoder struct {
	dataShards   int
	parityShards int
	field        *GF16
	// gen is the systematic (dataShards+parityShards) x dataShards generator
	// matrix: the top dataShards rows are the identity and the bottom
	// parityShards rows form a Cauchy matrix (MDS: every dataShards-row
	// sub-matrix is invertible).
	gen matrix
}

// New returns an encoder for dataShards data shards plus parityShards parity
// shards. It returns ErrInvalidParams if either count is non-positive or if
// dataShards+parityShards exceeds 65535.
func New(dataShards, parityShards int) (*Encoder, error) {
	if dataShards <= 0 || parityShards <= 0 || dataShards+parityShards > gf16Nonzero {
		return nil, ErrInvalidParams
	}
	f := NewGF16()
	total := dataShards + parityShards
	gen := newMatrix(total, dataShards)
	// Top block: identity, so data shards pass through unchanged (systematic).
	for i := 0; i < dataShards; i++ {
		gen[i][i] = 1
	}
	// Bottom block: Cauchy matrix c[i][j] = 1 / (x_i + y_j) with the parity
	// nodes x_i and data nodes y_j drawn from disjoint sets of distinct field
	// elements, guaranteeing x_i + y_j != 0 and the MDS property.
	for i := 0; i < parityShards; i++ {
		xi := uint16(dataShards + i)
		for j := 0; j < dataShards; j++ {
			yj := uint16(j)
			gen[dataShards+i][j] = f.Div(1, f.Add(xi, yj))
		}
	}
	return &Encoder{
		dataShards:   dataShards,
		parityShards: parityShards,
		field:        f,
		gen:          gen,
	}, nil
}

// DataShards returns the number of data shards.
func (e *Encoder) DataShards() int { return e.dataShards }

// ParityShards returns the number of parity shards.
func (e *Encoder) ParityShards() int { return e.parityShards }

// Field returns the GF(2^16) arithmetic core, so a PAR2 layer can reuse it.
func (e *Encoder) Field() *GF16 { return e.field }

// checkShards verifies every shard has the same even length and returns that
// length. shards must be non-empty.
func checkShards(shards [][]byte) (int, error) {
	size := len(shards[0])
	for _, s := range shards {
		if len(s) != size {
			return 0, ErrShardSize
		}
	}
	if size%2 != 0 {
		return 0, ErrShardSize
	}
	return size, nil
}

// Encode fills the parity shards from the data shards. shards must have length
// dataShards+parityShards; the first dataShards are read and the remaining
// parityShards are written.
func (e *Encoder) Encode(shards [][]byte) error {
	if len(shards) != e.dataShards+e.parityShards {
		return ErrShardCount
	}
	if _, err := checkShards(shards); err != nil {
		return err
	}
	for i := 0; i < e.parityShards; i++ {
		row := e.gen[e.dataShards+i]
		dst := shards[e.dataShards+i]
		galMul(e.field, dst, shards[0], row[0])
		for j := 1; j < e.dataShards; j++ {
			galMulAdd(e.field, dst, shards[j], row[j])
		}
	}
	return nil
}

// Verify reports whether the parity shards are consistent with the data shards.
func (e *Encoder) Verify(shards [][]byte) (bool, error) {
	if len(shards) != e.dataShards+e.parityShards {
		return false, ErrShardCount
	}
	size, err := checkShards(shards)
	if err != nil {
		return false, err
	}
	scratch := make([]byte, size)
	for i := 0; i < e.parityShards; i++ {
		row := e.gen[e.dataShards+i]
		galMul(e.field, scratch, shards[0], row[0])
		for j := 1; j < e.dataShards; j++ {
			galMulAdd(e.field, scratch, shards[j], row[j])
		}
		if !bytes.Equal(scratch, shards[e.dataShards+i]) {
			return false, nil
		}
	}
	return true, nil
}

// Reconstruct recovers missing shards in place. present[i] == false marks shard
// i (data or parity) as erased; each erased shard's slice must already be
// allocated to the common shard length. It succeeds when at least dataShards
// shards are present.
func (e *Encoder) Reconstruct(shards [][]byte, present []bool) error {
	total := e.dataShards + e.parityShards
	if len(shards) != total || len(present) != total {
		return ErrShardCount
	}
	if _, err := checkShards(shards); err != nil {
		return err
	}
	use := make([]int, 0, e.dataShards)
	for i := 0; i < total; i++ {
		if present[i] {
			use = append(use, i)
		}
	}
	if len(use) < e.dataShards {
		return ErrTooFewShards
	}
	use = use[:e.dataShards]

	// Build the sub-matrix of generator rows for the used present shards and
	// invert it; inv maps present-shard words back to the original data words.
	sub := newMatrix(e.dataShards, e.dataShards)
	for r, idx := range use {
		copy(sub[r], e.gen[idx])
	}
	inv := e.field.invert(sub)

	// Recover any missing data shards from the present shards.
	for j := 0; j < e.dataShards; j++ {
		if present[j] {
			continue
		}
		dst := shards[j]
		galMul(e.field, dst, shards[use[0]], inv[j][0])
		for r := 1; r < e.dataShards; r++ {
			galMulAdd(e.field, dst, shards[use[r]], inv[j][r])
		}
	}

	// Recompute any missing parity shards from the now-complete data shards.
	for i := 0; i < e.parityShards; i++ {
		if present[e.dataShards+i] {
			continue
		}
		row := e.gen[e.dataShards+i]
		dst := shards[e.dataShards+i]
		galMul(e.field, dst, shards[0], row[0])
		for j := 1; j < e.dataShards; j++ {
			galMulAdd(e.field, dst, shards[j], row[j])
		}
	}
	return nil
}
