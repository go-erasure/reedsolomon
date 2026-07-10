package reedsolomon

import (
	"bytes"
	"testing"
)

// makeShards builds dataShards+parityShards shards of the given byte length,
// filling the data shards with a deterministic pattern and zeroing parity.
func makeShards(t *testing.T, e *Encoder, shardLen int) [][]byte {
	t.Helper()
	total := e.dataShards + e.parityShards
	shards := make([][]byte, total)
	for i := range shards {
		shards[i] = make([]byte, shardLen)
	}
	seed := byte(1)
	for i := 0; i < e.dataShards; i++ {
		for k := range shards[i] {
			seed = seed*31 + byte(i*7+k)
			shards[i][k] = seed
		}
	}
	return shards
}

func cloneShards(src [][]byte) [][]byte {
	out := make([][]byte, len(src))
	for i, s := range src {
		out[i] = append([]byte(nil), s...)
	}
	return out
}

func TestNewErrors(t *testing.T) {
	cases := [][2]int{
		{0, 2},
		{2, 0},
		{-1, 2},
		{2, -1},
		{40000, 40000}, // sum > 65535
	}
	for _, c := range cases {
		if _, err := New(c[0], c[1]); err != ErrInvalidParams {
			t.Fatalf("New(%d,%d) err = %v, want ErrInvalidParams", c[0], c[1], err)
		}
	}
}

func TestNewAccessors(t *testing.T) {
	e, err := New(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if e.DataShards() != 4 || e.ParityShards() != 3 {
		t.Fatal("accessors wrong")
	}
	if e.Field() == nil {
		t.Fatal("Field() nil")
	}
}

func TestEncodeVerifyRoundTrip(t *testing.T) {
	e, err := New(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	shards := makeShards(t, e, 32)
	if err := e.Encode(shards); err != nil {
		t.Fatal(err)
	}
	ok, err := e.Verify(shards)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("Verify false after Encode")
	}
	// Corrupt a parity shard -> Verify must report false.
	shards[e.dataShards][0] ^= 0xFF
	ok, err = e.Verify(shards)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("Verify true after corruption")
	}
}

func TestEncodeErrors(t *testing.T) {
	e, _ := New(2, 2)
	// Wrong shard count.
	if err := e.Encode(make([][]byte, 3)); err != ErrShardCount {
		t.Fatalf("count err = %v", err)
	}
	// Unequal length.
	sh := [][]byte{make([]byte, 4), make([]byte, 2), make([]byte, 4), make([]byte, 4)}
	if err := e.Encode(sh); err != ErrShardSize {
		t.Fatalf("unequal err = %v", err)
	}
	// Odd length.
	odd := [][]byte{make([]byte, 3), make([]byte, 3), make([]byte, 3), make([]byte, 3)}
	if err := e.Encode(odd); err != ErrShardSize {
		t.Fatalf("odd err = %v", err)
	}
}

func TestVerifyErrors(t *testing.T) {
	e, _ := New(2, 2)
	if _, err := e.Verify(make([][]byte, 1)); err != ErrShardCount {
		t.Fatalf("count err = %v", err)
	}
	sh := [][]byte{make([]byte, 4), make([]byte, 2), make([]byte, 4), make([]byte, 4)}
	if _, err := e.Verify(sh); err != ErrShardSize {
		t.Fatalf("size err = %v", err)
	}
}

func TestReconstructErrors(t *testing.T) {
	e, _ := New(2, 2)
	shards := makeShards(t, e, 8)
	_ = e.Encode(shards)

	// Wrong shard count.
	if err := e.Reconstruct(make([][]byte, 3), make([]bool, 4)); err != ErrShardCount {
		t.Fatalf("shard count err = %v", err)
	}
	// Wrong present count.
	if err := e.Reconstruct(shards, make([]bool, 3)); err != ErrShardCount {
		t.Fatalf("present count err = %v", err)
	}
	// Bad shard size.
	bad := cloneShards(shards)
	bad[0] = make([]byte, 6)
	if err := e.Reconstruct(bad, []bool{true, true, true, true}); err != ErrShardSize {
		t.Fatalf("size err = %v", err)
	}
	// Too few shards present (only 1 of 2 data required).
	present := []bool{true, false, false, false}
	if err := e.Reconstruct(shards, present); err != ErrTooFewShards {
		t.Fatalf("too few err = %v", err)
	}
}

func eraseAndReconstruct(t *testing.T, e *Encoder, orig [][]byte, erase []int) {
	t.Helper()
	total := e.dataShards + e.parityShards
	work := cloneShards(orig)
	present := make([]bool, total)
	for i := range present {
		present[i] = true
	}
	for _, idx := range erase {
		present[idx] = false
		for k := range work[idx] {
			work[idx][k] = 0
		}
	}
	if err := e.Reconstruct(work, present); err != nil {
		t.Fatalf("Reconstruct(erase=%v) failed: %v", erase, err)
	}
	for i := range orig {
		if !bytes.Equal(work[i], orig[i]) {
			t.Fatalf("shard %d mismatch after reconstruct (erase=%v)", i, erase)
		}
	}
}

func TestReconstructScenarios(t *testing.T) {
	e, err := New(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	orig := makeShards(t, e, 64)
	if err := e.Encode(orig); err != nil {
		t.Fatal(err)
	}

	scenarios := [][]int{
		{1},       // single data shard
		{0},       // data shard 0 (forces pivot row swap)
		{4},       // single parity shard
		{5, 6},    // multiple parity shards only
		{0, 2, 5}, // mixed data + parity, includes shard 0
		{0, 1, 2}, // maximal data erasure (3 = parityShards)
		{},        // nothing erased (no-op recovery)
	}
	for _, s := range scenarios {
		eraseAndReconstruct(t, e, orig, s)
	}
}

func TestReconstructSingleDataSingleParity(t *testing.T) {
	// Exercises dataShards == 1 (galMul only, no galMulAdd in the row loops).
	e, err := New(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	orig := makeShards(t, e, 16)
	if err := e.Encode(orig); err != nil {
		t.Fatal(err)
	}
	eraseAndReconstruct(t, e, orig, []int{0})
	eraseAndReconstruct(t, e, orig, []int{1})
}
