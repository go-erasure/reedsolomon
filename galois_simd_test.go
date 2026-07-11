package reedsolomon

import (
	"bytes"
	"math/rand"
	"testing"
)

// The SIMD split-table kernels (galois_amd64.s / galois_arm64.s) are validated
// here against the scalar oracle (galMulScalar / galMulAddScalar) bit for bit.
// On architectures without a kernel galMul/galMulAdd are the scalar loops
// themselves, so these tests still pass (scalar vs scalar) and keep the dispatch
// covered everywhere.

// diffLengths spans the interesting boundaries of the 32-byte SIMD block: empty,
// a single word, sub-block, exactly one/two blocks, blocks plus an odd-word tail,
// and large regions that exercise the inner loop many times.
var diffLengths = []int{0, 2, 4, 6, 30, 32, 34, 46, 62, 64, 66, 96, 128, 130, 1024, 4096, 65536}

// checkMul asserts galMul(dst,src,coeff) equals an independent scalar multiply
// into a fresh buffer, byte for byte.
func checkMul(t *testing.T, f *GF16, src []byte, coeff uint16) {
	t.Helper()
	got := make([]byte, len(src))
	galMul(f, got, src, coeff)
	want := make([]byte, len(src))
	galMulScalar(f, want, src, coeff)
	if !bytes.Equal(got, want) {
		t.Fatalf("galMul len=%d coeff=%#x: SIMD != scalar", len(src), coeff)
	}
}

// checkMulAdd asserts galMulAdd(dst,src,coeff) equals the scalar accumulate over
// the same initial dst, byte for byte.
func checkMulAdd(t *testing.T, f *GF16, dst, src []byte, coeff uint16) {
	t.Helper()
	got := append([]byte(nil), dst...)
	galMulAdd(f, got, src, coeff)
	want := append([]byte(nil), dst...)
	galMulAddScalar(f, want, src, coeff)
	if !bytes.Equal(got, want) {
		t.Fatalf("galMulAdd len=%d coeff=%#x: SIMD != scalar", len(src), coeff)
	}
}

func TestGalMulDifferential(t *testing.T) {
	f := NewGF16()
	r := rand.New(rand.NewSource(1))
	// coeff 0 and 1 are the field's absorbing/identity elements; the rest are
	// random including values that stress every nibble table.
	coeffs := []uint16{0, 1, 2, 0x1234, 0x8000, 0xFFFF, 0xABCD, 0x00FF, 0xFF00}
	for i := 0; i < 32; i++ {
		coeffs = append(coeffs, uint16(r.Intn(1<<16)))
	}
	for _, n := range diffLengths {
		src := make([]byte, n)
		r.Read(src)
		dst := make([]byte, n)
		r.Read(dst)
		for _, c := range coeffs {
			checkMul(t, f, src, c)
			checkMulAdd(t, f, dst, src, c)
		}
	}
}

// TestGalMulOddTail exercises an odd-length buffer (trailing byte ignored by the
// word loop) to match the scalar oracle's boundary exactly.
func TestGalMulOddTail(t *testing.T) {
	f := NewGF16()
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 3, 33, 65} {
		src := make([]byte, n)
		r.Read(src)
		dst := make([]byte, n)
		r.Read(dst)
		checkMul(t, f, src, 0x2718)
		checkMulAdd(t, f, dst, src, 0x3141)
	}
}

// FuzzGalMul drives random buffers and coefficients through both kernels and the
// scalar oracle; any divergence fails. Seeds cover the block boundaries.
func FuzzGalMul(f *testing.F) {
	f.Add([]byte{0x12, 0x34, 0xAB, 0xCD}, uint16(7))
	f.Add(make([]byte, 64), uint16(0xFFFF))
	f.Add(make([]byte, 66), uint16(0))
	f.Add(make([]byte, 130), uint16(1))
	field := NewGF16()
	f.Fuzz(func(t *testing.T, data []byte, coeff uint16) {
		src := data
		dst := make([]byte, len(src))
		if len(src) > 0 {
			for i := range dst {
				dst[i] = src[len(src)-1-i]
			}
		}
		got := make([]byte, len(src))
		galMul(field, got, src, coeff)
		want := make([]byte, len(src))
		galMulScalar(field, want, src, coeff)
		if !bytes.Equal(got, want) {
			t.Fatalf("galMul mismatch len=%d coeff=%#x", len(src), coeff)
		}
		gotA := append([]byte(nil), dst...)
		galMulAdd(field, gotA, src, coeff)
		wantA := append([]byte(nil), dst...)
		galMulAddScalar(field, wantA, src, coeff)
		if !bytes.Equal(gotA, wantA) {
			t.Fatalf("galMulAdd mismatch len=%d coeff=%#x", len(src), coeff)
		}
	})
}

// benchRegion is a representative shard size (1 MiB) for the region benchmarks.
const benchRegion = 1 << 20

func BenchmarkGalMulAddScalar(b *testing.B) {
	f := NewGF16()
	src := make([]byte, benchRegion)
	rand.New(rand.NewSource(3)).Read(src)
	dst := make([]byte, benchRegion)
	b.SetBytes(benchRegion)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		galMulAddScalar(f, dst, src, 0x9A3F)
	}
}

func BenchmarkGalMulAddSIMD(b *testing.B) {
	f := NewGF16()
	src := make([]byte, benchRegion)
	rand.New(rand.NewSource(3)).Read(src)
	dst := make([]byte, benchRegion)
	b.SetBytes(benchRegion)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		galMulAdd(f, dst, src, 0x9A3F)
	}
}
