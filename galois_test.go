package reedsolomon

import "testing"

func TestGalMul(t *testing.T) {
	f := NewGF16()
	src := []byte{0x12, 0x34, 0xAB, 0xCD}
	dst := make([]byte, 4)
	coeff := uint16(0x0007)
	galMul(f, dst, src, coeff)
	for w := 0; w < 2; w++ {
		v := uint16(src[2*w])<<8 | uint16(src[2*w+1])
		want := f.Mul(coeff, v)
		got := uint16(dst[2*w])<<8 | uint16(dst[2*w+1])
		if got != want {
			t.Fatalf("galMul word %d = %#x, want %#x", w, got, want)
		}
	}
}

func TestGalMulAdd(t *testing.T) {
	f := NewGF16()
	src := []byte{0x12, 0x34, 0xAB, 0xCD}
	dst := []byte{0x00, 0xFF, 0x55, 0xAA}
	orig := append([]byte(nil), dst...)
	coeff := uint16(0x00F1)
	galMulAdd(f, dst, src, coeff)
	for w := 0; w < 2; w++ {
		v := uint16(src[2*w])<<8 | uint16(src[2*w+1])
		d := uint16(orig[2*w])<<8 | uint16(orig[2*w+1])
		want := d ^ f.Mul(coeff, v)
		got := uint16(dst[2*w])<<8 | uint16(dst[2*w+1])
		if got != want {
			t.Fatalf("galMulAdd word %d = %#x, want %#x", w, got, want)
		}
	}
}
