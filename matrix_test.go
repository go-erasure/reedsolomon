package reedsolomon

import "testing"

func TestInvertIdentity(t *testing.T) {
	f := NewGF16()
	m := matrix{{1, 0}, {0, 1}}
	inv := f.invert(m)
	if inv[0][0] != 1 || inv[0][1] != 0 || inv[1][0] != 0 || inv[1][1] != 1 {
		t.Fatalf("identity inverse wrong: %v", inv)
	}
}

func TestInvertRoundTrip(t *testing.T) {
	f := NewGF16()
	// A matrix whose first pivot is zero, forcing a row swap.
	m := matrix{
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 9},
	}
	inv := f.invert(m)
	// m * inv must be the identity.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var acc uint16
			for k := 0; k < 3; k++ {
				acc ^= f.Mul(m[i][k], inv[k][j])
			}
			want := uint16(0)
			if i == j {
				want = 1
			}
			if acc != want {
				t.Fatalf("m*inv[%d][%d] = %#x, want %#x", i, j, acc, want)
			}
		}
	}
}

func TestInvertSingularPanics(t *testing.T) {
	f := NewGF16()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("inverting singular matrix did not panic")
		}
	}()
	// Two identical rows -> singular.
	_ = f.invert(matrix{{1, 1}, {1, 1}})
}
