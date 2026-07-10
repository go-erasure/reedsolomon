package reedsolomon

import "testing"

func TestGF16TablesConsistent(t *testing.T) {
	f := NewGF16()
	// exp and log are inverses over the whole multiplicative group.
	for i := 0; i < gf16Nonzero; i++ {
		x := f.exp[i]
		if int(f.log[x]) != i {
			t.Fatalf("log[exp[%d]] = %d, want %d", i, f.log[x], i)
		}
	}
	// The doubled half of exp mirrors the first half.
	for i := 0; i < gf16Nonzero; i++ {
		if f.exp[i] != f.exp[i+gf16Nonzero] {
			t.Fatalf("exp not periodic at %d", i)
		}
	}
}

func TestGF16Add(t *testing.T) {
	f := NewGF16()
	if got := f.Add(0xABCD, 0x1234); got != 0xABCD^0x1234 {
		t.Fatalf("Add = %#x", got)
	}
	if f.Add(0x5555, 0x5555) != 0 {
		t.Fatal("a+a must be 0")
	}
}

func TestGF16MulZero(t *testing.T) {
	f := NewGF16()
	if f.Mul(0, 0x1234) != 0 {
		t.Fatal("0*x != 0")
	}
	if f.Mul(0x1234, 0) != 0 {
		t.Fatal("x*0 != 0")
	}
}

func TestGF16MulCommutativeAndIdentity(t *testing.T) {
	f := NewGF16()
	for _, a := range []uint16{1, 2, 0x1234, 0xFFFF, 0x8000} {
		if f.Mul(a, 1) != a {
			t.Fatalf("Mul(%#x,1) != a", a)
		}
		for _, b := range []uint16{3, 0x00FF, 0xBEEF} {
			if f.Mul(a, b) != f.Mul(b, a) {
				t.Fatalf("Mul not commutative for %#x,%#x", a, b)
			}
		}
	}
}

func TestGF16DivRoundTrip(t *testing.T) {
	f := NewGF16()
	for _, a := range []uint16{0, 1, 2, 0x1234, 0xFFFF} {
		for _, b := range []uint16{1, 3, 0x00FF, 0xBEEF, 0xFFFF} {
			q := f.Div(a, b)
			if f.Mul(q, b) != a {
				t.Fatalf("Div/Mul round-trip failed for a=%#x b=%#x", a, b)
			}
		}
	}
}

func TestGF16DivZeroNumerator(t *testing.T) {
	f := NewGF16()
	if f.Div(0, 0x1234) != 0 {
		t.Fatal("0/x must be 0")
	}
}

func TestGF16DivByZeroPanics(t *testing.T) {
	f := NewGF16()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Div by zero did not panic")
		}
	}()
	_ = f.Div(1, 0)
}

func TestGF16Exp(t *testing.T) {
	f := NewGF16()
	if f.Exp(0) != 1 {
		t.Fatal("g^0 must be 1")
	}
	if f.Exp(1) != 2 {
		t.Fatal("g^1 must be 2 (generator)")
	}
	// Reduction of large and negative exponents modulo the group order.
	if f.Exp(gf16Nonzero) != f.Exp(0) {
		t.Fatal("g^order must wrap to g^0")
	}
	if f.Exp(-1) != f.Exp(gf16Nonzero-1) {
		t.Fatal("negative exponent must wrap")
	}
	// g^log(x) == x for a few values.
	for _, x := range []uint16{2, 0x1234, 0xFFFF} {
		if f.Exp(int(f.Log(x))) != x {
			t.Fatalf("Exp(Log(%#x)) != x", x)
		}
	}
}

func TestGF16Pow(t *testing.T) {
	f := NewGF16()
	if f.Pow(0x1234, 0) != 1 {
		t.Fatal("a^0 must be 1")
	}
	if f.Pow(0, 0) != 1 {
		t.Fatal("0^0 defined as 1")
	}
	if f.Pow(0, 5) != 0 {
		t.Fatal("0^n must be 0 for n>0")
	}
	a := uint16(0x1234)
	// a^2 == a*a.
	if f.Pow(a, 2) != f.Mul(a, a) {
		t.Fatal("Pow(a,2) != a*a")
	}
	// a^3 == a*a*a.
	if f.Pow(a, 3) != f.Mul(f.Mul(a, a), a) {
		t.Fatal("Pow(a,3) mismatch")
	}
	// Negative exponent: a^-1 * a == 1.
	if f.Mul(f.Pow(a, -1), a) != 1 {
		t.Fatal("a^-1 * a != 1")
	}
	// Exponent larger than the group order must wrap.
	if f.Pow(a, gf16Nonzero+1) != a {
		t.Fatalf("a^(order+1) != a")
	}
}

func TestGF16Log(t *testing.T) {
	f := NewGF16()
	if f.Log(1) != 0 {
		t.Fatal("log(1) must be 0")
	}
	if f.Log(2) != 1 {
		t.Fatal("log(generator) must be 1")
	}
}
