package reedsolomon

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// The oracle is par2cmdline v2.0.0 (see testdata/par2/README.md): the .par2
// files in testdata were written by it, so their recovery slices are what a
// conforming implementation must produce, and what it must be able to decode.

var (
	par2Magic    = []byte("PAR2\x00PKT")
	par2TypeMain = []byte("PAR 2.0\x00Main\x00\x00\x00\x00")
	par2TypeFile = []byte("PAR 2.0\x00FileDesc")
	par2TypeRecv = []byte("PAR 2.0\x00RecvSlic")
)

// par2Set is what the test needs out of a PAR2 file set: the data slices in
// recovery-set order, and the recovery slices with their exponents.
type par2Set struct {
	sliceSize int
	data      [][]byte
	exponents []uint16
	recovery  [][]byte
}

// loadPAR2 parses every .par2 file of dir (packet layout: PAR 2.0 specification,
// "Packet Layout") and reads the input files they describe from the same
// directory, splitting them into slices and zero-padding the last one.
func loadPAR2(t testing.TB, dir string) par2Set {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.par2"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no .par2 files in %s: %v", dir, err)
	}
	var (
		set      par2Set
		order    [][16]byte
		names    = map[[16]byte]string{}
		seenExps = map[uint16]bool{}
	)
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for len(raw) > 0 {
			if len(raw) < 64 || !bytes.Equal(raw[:8], par2Magic) {
				t.Fatalf("%s: bad packet header", path)
			}
			n := int(binary.LittleEndian.Uint64(raw[8:16]))
			if n < 64 || n > len(raw) || n%4 != 0 {
				t.Fatalf("%s: bad packet length %d", path, n)
			}
			typ, body := raw[48:64], raw[64:n]
			raw = raw[n:]
			switch {
			case bytes.Equal(typ, par2TypeMain):
				set.sliceSize = int(binary.LittleEndian.Uint64(body[:8]))
				count := int(binary.LittleEndian.Uint32(body[8:12]))
				order = order[:0]
				for i := 0; i < count; i++ {
					var id [16]byte
					copy(id[:], body[12+16*i:])
					order = append(order, id)
				}
			case bytes.Equal(typ, par2TypeFile):
				var id [16]byte
				copy(id[:], body[:16])
				names[id] = string(bytes.TrimRight(body[56:], "\x00"))
			case bytes.Equal(typ, par2TypeRecv):
				e := uint16(binary.LittleEndian.Uint32(body[:4]))
				if !seenExps[e] {
					seenExps[e] = true
					set.exponents = append(set.exponents, e)
					set.recovery = append(set.recovery, append([]byte(nil), body[4:]...))
				}
			}
		}
	}
	for _, id := range order {
		content, err := os.ReadFile(filepath.Join(dir, names[id]))
		if err != nil {
			t.Fatal(err)
		}
		for off := 0; off < len(content); off += set.sliceSize {
			s := make([]byte, set.sliceSize)
			copy(s, content[off:])
			set.data = append(set.data, s)
		}
	}
	// Order recovery slices by exponent so the test is independent of file order.
	idx := make([]int, len(set.exponents))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return set.exponents[idx[a]] < set.exponents[idx[b]] })
	exps, recs := make([]uint16, len(idx)), make([][]byte, len(idx))
	for k, i := range idx {
		exps[k], recs[k] = set.exponents[i], set.recovery[i]
	}
	set.exponents, set.recovery = exps, recs
	return set
}

var par2Fixtures = []string{"testdata/par2/single", "testdata/par2/multi"}

func clone(s [][]byte) [][]byte {
	o := make([][]byte, len(s))
	for i := range s {
		o[i] = append([]byte(nil), s[i]...)
	}
	return o
}

// (a) Our recovery slices are byte-identical to par2cmdline's.
func TestPAR2MatchesPar2cmdline(t *testing.T) {
	for _, dir := range par2Fixtures {
		t.Run(dir, func(t *testing.T) {
			set := loadPAR2(t, dir)
			if len(set.data) < 2 || len(set.recovery) < 2 {
				t.Fatalf("degenerate fixture: %d data, %d recovery", len(set.data), len(set.recovery))
			}
			p, err := NewPAR2(len(set.data))
			if err != nil {
				t.Fatal(err)
			}
			got := make([][]byte, len(set.recovery))
			for i := range got {
				got[i] = make([]byte, set.sliceSize)
			}
			if err := p.Encode(set.data, set.exponents, got); err != nil {
				t.Fatal(err)
			}
			for k := range got {
				if !bytes.Equal(got[k], set.recovery[k]) {
					t.Fatalf("recovery slice of exponent %d differs from par2cmdline's", set.exponents[k])
				}
			}
			ok, err := p.Verify(set.data, set.exponents, set.recovery)
			if err != nil || !ok {
				t.Fatalf("Verify of par2cmdline's recovery data = %v, %v", ok, err)
			}
			set.recovery[0][0] ^= 1
			if ok, _ := p.Verify(set.data, set.exponents, set.recovery); ok {
				t.Fatal("Verify accepted a corrupted recovery slice")
			}
		})
	}
}

// (b) Deleted data slices are rebuilt from par2cmdline's recovery data: every
// single loss, every pair, and random larger losses up to the recovery count.
func TestPAR2ReconstructsFromPar2cmdline(t *testing.T) {
	for _, dir := range par2Fixtures {
		t.Run(dir, func(t *testing.T) {
			set := loadPAR2(t, dir)
			n, nrec := len(set.data), len(set.recovery)
			p, _ := NewPAR2(n)
			try := func(lost []int) {
				t.Helper()
				data := clone(set.data)
				present := make([]bool, n)
				for i := range present {
					present[i] = true
				}
				for _, i := range lost {
					present[i] = false
					clear(data[i])
				}
				if err := p.Reconstruct(data, present, set.exponents, set.recovery); err != nil {
					t.Fatalf("lost %v: %v", lost, err)
				}
				for i := range data {
					if !bytes.Equal(data[i], set.data[i]) {
						t.Fatalf("lost %v: slice %d not restored", lost, i)
					}
				}
			}
			for i := 0; i < n; i++ {
				try([]int{i})
				for j := i + 1; j < n; j++ {
					try([]int{i, j})
				}
			}
			r := rand.New(rand.NewSource(1))
			for iter := 0; iter < 200; iter++ {
				try(r.Perm(n)[:1+r.Intn(nrec)])
			}
			// One loss too many is refused, not guessed.
			data := clone(set.data)
			present := make([]bool, n)
			for i := range present {
				present[i] = i > nrec
			}
			if err := p.Reconstruct(data, present, set.exponents, set.recovery); !errors.Is(err, ErrTooFewShards) {
				t.Fatalf("losing %d slices with %d recovery slices: %v", nrec+1, nrec, err)
			}
		})
	}
}

// The constants are pinned independently of NewPAR2: par2cmdline's rule is
// "the i-th logbase coprime to 65535", i.e. powers of two with order 65535
// whose exponent is not 0 modulo 3, 5, 17 or 257.
func TestPAR2Constants(t *testing.T) {
	p, err := NewPAR2(MaxPAR2DataSlices)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint16{2, 4, 16, 128, 256, 2048} { // 2^1, 2^2, 2^4, 2^7, 2^8, 2^11
		if got := p.Constant(i); got != want {
			t.Fatalf("Constant(%d) = %d, want %d", i, got, want)
		}
	}
	f := NewGF16()
	x, seen, i := uint16(1), map[uint16]bool{}, 0
	for k := 1; k < 65535; k++ { // x = 2^k
		x = f.Mul(x, 2)
		if k%3 == 0 || k%5 == 0 || k%17 == 0 || k%257 == 0 {
			continue
		}
		if p.Constant(i) != x || p.logs[i] != uint16(k) {
			t.Fatalf("Constant(%d) = %d (log %d), want 2^%d = %d", i, p.Constant(i), p.logs[i], k, x)
		}
		if seen[x] {
			t.Fatalf("constant %d repeated", x)
		}
		seen[x] = true
		i++
	}
	if i != MaxPAR2DataSlices {
		t.Fatalf("%d constants, want %d", i, MaxPAR2DataSlices)
	}
}

// try runs f and fails the test if it panics: a rejected input must come back
// as an error from the guard, not as a crash further in.
func try(t *testing.T, f func() error) (err error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panic instead of an error: %v", r)
		}
	}()
	return f()
}

func TestPAR2Params(t *testing.T) {
	for _, n := range []int{0, -1, MaxPAR2DataSlices + 1} {
		if _, err := NewPAR2(n); !errors.Is(err, ErrInvalidParams) {
			t.Fatalf("NewPAR2(%d) = %v", n, err)
		}
	}
	p, _ := NewPAR2(3)
	if p.DataSlices() != 3 {
		t.Fatal("DataSlices")
	}
	good := [][]byte{make([]byte, 4), make([]byte, 4), make([]byte, 4)}
	rec := [][]byte{make([]byte, 4)}
	if try(t, func() error { return p.Encode(good[:2], []uint16{0}, rec) }) != ErrShardCount ||
		try(t, func() error { return p.Encode(good, []uint16{0, 1}, rec) }) != ErrShardCount {
		t.Fatal("count mismatch accepted")
	}
	if err := try(t, func() error {
		return p.Encode([][]byte{make([]byte, 4), make([]byte, 6), make([]byte, 4)}, []uint16{0}, rec)
	}); err != ErrShardSize {
		t.Fatalf("unequal sizes: %v", err)
	}
	if err := try(t, func() error {
		return p.Encode([][]byte{make([]byte, 3), make([]byte, 3), make([]byte, 3)}, []uint16{0}, [][]byte{make([]byte, 3)})
	}); err != ErrShardSize {
		t.Fatalf("odd sizes: %v", err)
	}
	if err := try(t, func() error { _, err := p.Verify(good, []uint16{0}, nil); return err }); err != ErrShardCount {
		t.Fatalf("Verify count mismatch: %v", err)
	}
	if err := p.Reconstruct(good, []bool{true}, nil, nil); err != ErrShardCount {
		t.Fatalf("Reconstruct present mismatch: %v", err)
	}
	if err := p.Reconstruct(good, []bool{true, true, true}, nil, nil); err != nil {
		t.Fatalf("nothing missing: %v", err)
	}
	// Nothing missing is a no-op: recovery slices are not even looked at.
	if err := p.Reconstruct(good, []bool{true, true, true}, []uint16{0}, [][]byte{make([]byte, 3)}); err != nil {
		t.Fatalf("nothing missing, odd recovery slice: %v", err)
	}
	if ok, err := p.Verify(good, nil, nil); !ok || err != nil {
		t.Fatalf("Verify with no recovery: %v %v", ok, err)
	}
	if err := p.Encode(good, nil, nil); err != nil {
		t.Fatalf("Encode with no recovery: %v", err)
	}

	// The same checks on the other entry points: wrong slice count, odd size.
	odd := [][]byte{make([]byte, 3), make([]byte, 3), make([]byte, 3)}
	oddRec := [][]byte{make([]byte, 3)}
	if err := try(t, func() error { _, err := p.Verify(good[:2], []uint16{0}, rec); return err }); err != ErrShardCount {
		t.Fatalf("Verify wrong data count: %v", err)
	}
	if err := try(t, func() error { _, err := p.Verify(odd, []uint16{0}, oddRec); return err }); err != ErrShardSize {
		t.Fatalf("Verify odd size: %v", err)
	}
	if err := p.Reconstruct(good[:2], []bool{true, true}, nil, nil); err != ErrShardCount {
		t.Fatalf("Reconstruct wrong data count: %v", err)
	}
	if err := try(t, func() error { return p.Reconstruct(odd, []bool{false, true, true}, []uint16{0}, oddRec) }); err != ErrShardSize {
		t.Fatalf("Reconstruct odd size: %v", err)
	}
}

// The Vandermonde code can be singular for unlucky exponents; surplus recovery
// slices stand in, duplicates never help.
func TestPAR2SingularAndSurplus(t *testing.T) {
	const n, size = 6, 64
	p, _ := NewPAR2(n)
	r := rand.New(rand.NewSource(3))
	data := make([][]byte, n)
	for i := range data {
		data[i] = make([]byte, size)
		r.Read(data[i])
	}
	mk := func(exps []uint16) [][]byte {
		rec := make([][]byte, len(exps))
		for i := range rec {
			rec[i] = make([]byte, size)
		}
		if err := p.Encode(data, exps, rec); err != nil {
			t.Fatal(err)
		}
		return rec
	}
	lose := func() ([][]byte, []bool) {
		d, present := clone(data), []bool{true, false, true, false, true, true}
		clear(d[1])
		clear(d[3])
		return d, present
	}
	// 0 and 65535 are the same exponent (2^65535 = 1): dependent rows.
	exps := []uint16{0, 65535, 4}
	d, present := lose()
	if err := p.Reconstruct(d, present, exps, mk(exps)); err != nil {
		t.Fatalf("surplus slice should replace the dependent one: %v", err)
	}
	if !bytes.Equal(d[1], data[1]) || !bytes.Equal(d[3], data[3]) {
		t.Fatal("not restored")
	}
	exps = []uint16{7, 7}
	d, present = lose()
	if err := try(t, func() error { return p.Reconstruct(d, present, exps, mk(exps)) }); !errors.Is(err, ErrSingular) {
		t.Fatalf("duplicate exponents: %v", err)
	}
}

// Self-consistency (not the oracle): random shapes, every tier of the kernels
// against the scalar LE oracle, and round trips through Reconstruct.
func TestPAR2RoundTripRandom(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	f := NewGF16()
	for iter := 0; iter < 60; iter++ {
		n := 1 + r.Intn(10)
		size := 2 * r.Intn(200)
		if size == 0 {
			size = 2
		}
		nrec := 1 + r.Intn(5)
		p, _ := NewPAR2(n)
		data := make([][]byte, n)
		for i := range data {
			data[i] = make([]byte, size)
			r.Read(data[i])
		}
		exps := make([]uint16, nrec)
		for k := range exps {
			exps[k] = uint16(r.Intn(1 << 16))
		}
		rec := make([][]byte, nrec)
		for k := range rec {
			rec[k] = make([]byte, size)
		}
		if err := p.Encode(data, exps, rec); err != nil {
			t.Fatal(err)
		}
		// The scalar little-endian oracle gives the same bytes.
		for k, e := range exps {
			want := make([]byte, size)
			for i := range data {
				galMulAddLEScalar(f, want, data[i], f.Pow(p.Constant(i), int(e)))
			}
			if !bytes.Equal(rec[k], want) {
				t.Fatalf("iter %d: recovery %d differs from the scalar oracle", iter, k)
			}
		}
		lost := r.Perm(n)[:r.Intn(min(n, nrec)+1)]
		d, present := clone(data), make([]bool, n)
		for i := range present {
			present[i] = true
		}
		for _, i := range lost {
			present[i] = false
			clear(d[i])
		}
		err := p.Reconstruct(d, present, exps, rec)
		if errors.Is(err, ErrSingular) {
			continue // legitimate for random exponents
		}
		if err != nil {
			t.Fatal(err)
		}
		for i := range d {
			if !bytes.Equal(d[i], data[i]) {
				t.Fatalf("iter %d: slice %d not restored", iter, i)
			}
		}
	}
}

// The kernel's LE entry points against the scalar LE oracle, all constants.
func TestLEKernelsMatchScalar(t *testing.T) {
	f := NewGF16()
	r := rand.New(rand.NewSource(5))
	src, dst := make([]byte, 130), make([]byte, 130)
	r.Read(src)
	r.Read(dst)
	for c := 0; c < 1<<16; c += 7 {
		got, want := append([]byte(nil), dst...), append([]byte(nil), dst...)
		galMulAddLE(f, got, src, uint16(c))
		galMulAddLEScalar(f, want, src, uint16(c))
		if !bytes.Equal(got, want) {
			t.Fatalf("galMulAddLE c=%#x", c)
		}
		galMulLE(f, got, src, uint16(c))
		galMulLEScalar(f, want, src, uint16(c))
		if !bytes.Equal(got, want) {
			t.Fatalf("galMulLE c=%#x", c)
		}
	}
}

// pickRows must select independent rows and report rank deficiency.
func TestPickRows(t *testing.T) {
	f := NewGF16()
	c := matrix{{1, 2}, {1, 2}, {3, 5}, {0, 0}}
	rows, inv, ok := f.pickRows(c, 2)
	if !ok || len(rows) != 2 || rows[0] != 0 || rows[1] != 2 || len(inv) != 2 {
		t.Fatalf("pickRows = %v %v %v", rows, inv, ok)
	}
	if _, _, ok := f.pickRows(matrix{{1, 2}, {1, 2}, {f.Mul(7, 1), f.Mul(7, 2)}}, 2); ok {
		t.Fatal("rank-1 matrix accepted")
	}
}

// FuzzPAR2Reconstruct fuzzes the decode path: arbitrary shapes, exponents and
// loss patterns must never panic, must restore the data or report ErrSingular /
// ErrTooFewShards, and arbitrary (corrupt) recovery bytes must be survivable.
func FuzzPAR2Reconstruct(f *testing.F) {
	f.Add([]byte("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), uint8(3), uint8(2), uint32(0b101), uint16(0))
	f.Add(bytes.Repeat([]byte{0xFF}, 64), uint8(0), uint8(0), uint32(1), uint16(65535))
	f.Add(make([]byte, 96), uint8(5), uint8(5), uint32(0x3F), uint16(40000))
	f.Add(bytes.Repeat([]byte{0x01, 0x80}, 50), uint8(7), uint8(1), uint32(0xAA), uint16(3))
	f.Fuzz(func(t *testing.T, raw []byte, nd, nr uint8, mask uint32, exp0 uint16) {
		n := 1 + int(nd)%12
		nrec := 1 + int(nr)%6
		size := len(raw) / n &^ 1
		if size == 0 {
			return
		}
		p, err := NewPAR2(n)
		if err != nil {
			t.Fatal(err)
		}
		data := make([][]byte, n)
		for i := range data {
			data[i] = append([]byte(nil), raw[i*size:(i+1)*size]...)
		}
		exps := make([]uint16, nrec)
		rec := make([][]byte, nrec)
		for k := range exps {
			exps[k] = exp0 + uint16(k)*uint16(1+mask>>24)
			rec[k] = make([]byte, size)
		}
		if err := p.Encode(data, exps, rec); err != nil {
			t.Fatal(err)
		}
		d, present := clone(data), make([]bool, n)
		for i := range present {
			present[i] = mask>>uint(i)&1 == 0
			if !present[i] {
				clear(d[i])
			}
		}
		switch err := p.Reconstruct(d, present, exps, rec); {
		case err == nil:
			for i := range d {
				if !bytes.Equal(d[i], data[i]) {
					t.Fatalf("slice %d not restored", i)
				}
			}
		case errors.Is(err, ErrSingular), errors.Is(err, ErrTooFewShards):
		default:
			t.Fatal(err)
		}
		// Corrupt recovery data: any outcome but a panic.
		for k := range rec {
			for j := range rec[k] {
				rec[k][j] ^= byte(j + k)
			}
		}
		_ = p.Reconstruct(clone(d), present, exps, rec)
	})
}
