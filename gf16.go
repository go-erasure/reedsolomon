// Package reedsolomon implements a pure-Go, dependency-free Reed-Solomon
// erasure code over GF(2^16).
//
// The finite field uses the primitive polynomial 0x1100B and generator 2,
// making it byte-for-byte compatible with the field used by the PAR2 recovery
// format. The GF16 type is exported so that a PAR2 layer built on top of this
// package can reuse the exact same arithmetic core.
package reedsolomon

const (
	// gf16Order is the number of elements in GF(2^16).
	gf16Order = 1 << 16
	// gf16Nonzero is the size of the multiplicative group (65535).
	gf16Nonzero = gf16Order - 1
	// gf16Poly is the primitive polynomial x^16 + x^12 + x^3 + x + 1 (0x1100B),
	// the PAR2-compatible reduction polynomial for GF(2^16).
	gf16Poly = 0x1100B
)

// GF16 provides arithmetic over GF(2^16) with primitive polynomial 0x1100B and
// generator 2. It holds precomputed exponent and logarithm tables over the
// 65535-element multiplicative group.
type GF16 struct {
	exp []uint16 // exp[i] = generator^i, doubled to length 2*65535 to avoid modulo
	log []uint16 // log[x] = discrete log of x base generator (log[0] is unused)
}

// NewGF16 returns a GF16 with its exp/log tables built for generator 2 and
// primitive polynomial 0x1100B.
func NewGF16() *GF16 {
	f := &GF16{
		exp: make([]uint16, 2*gf16Nonzero),
		log: make([]uint16, gf16Order),
	}
	x := 1
	for i := 0; i < gf16Nonzero; i++ {
		f.exp[i] = uint16(x)
		f.log[x] = uint16(i)
		x <<= 1 // multiply by the generator (2)
		if x&gf16Order != 0 {
			x ^= gf16Poly
		}
	}
	// Duplicate the table so exp[a+b] is valid for a,b in [0, 65534].
	for i := gf16Nonzero; i < 2*gf16Nonzero; i++ {
		f.exp[i] = f.exp[i-gf16Nonzero]
	}
	return f
}

// Add returns a + b in GF(2^16), which is the XOR of the two elements.
func (f *GF16) Add(a, b uint16) uint16 {
	return a ^ b
}

// Mul returns a * b in GF(2^16).
func (f *GF16) Mul(a, b uint16) uint16 {
	if a == 0 || b == 0 {
		return 0
	}
	return f.exp[int(f.log[a])+int(f.log[b])]
}

// Div returns a / b in GF(2^16). Dividing by zero panics; callers must ensure
// b is non-zero.
func (f *GF16) Div(a, b uint16) uint16 {
	if b == 0 {
		panic("reedsolomon: division by zero in GF(2^16)")
	}
	if a == 0 {
		return 0
	}
	return f.exp[int(f.log[a])-int(f.log[b])+gf16Nonzero]
}

// Exp returns generator^power. Negative and large powers are reduced modulo the
// order of the multiplicative group.
func (f *GF16) Exp(power int) uint16 {
	power %= gf16Nonzero
	if power < 0 {
		power += gf16Nonzero
	}
	return f.exp[power]
}

// Pow returns a raised to the n-th power in GF(2^16). Pow(0, 0) is defined as 1.
func (f *GF16) Pow(a uint16, n int) uint16 {
	if n == 0 {
		return 1
	}
	if a == 0 {
		return 0
	}
	l := (int(f.log[a]) * (n % gf16Nonzero)) % gf16Nonzero
	if l < 0 {
		l += gf16Nonzero
	}
	return f.exp[l]
}

// Log returns the discrete logarithm of a base the generator. Log(0) is
// undefined and returns 0.
func (f *GF16) Log(a uint16) uint16 {
	return f.log[a]
}
