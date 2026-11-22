package reedsolomon

import (
	"math/rand"
	"testing"
)

// benchShape: 100 data slices of 64 KiB, 20 recovery slices: a 6.4 MB set with
// 20% redundancy, the shape the par2cmdline comparison in the README uses
// (par2 create -s65536 -c20 over a 6553600-byte file). SetBytes is the data
// size, so MB/s is input throughput.
const (
	benchData  = 100
	benchPar   = 20
	benchSlice = 64 << 10
)

func benchSlices(n int) [][]byte {
	r := rand.New(rand.NewSource(1))
	s := make([][]byte, n)
	for i := range s {
		s[i] = make([]byte, benchSlice)
		r.Read(s[i])
	}
	return s
}

func BenchmarkEncodePAR2(b *testing.B) {
	p, _ := NewPAR2(benchData)
	data, rec := benchSlices(benchData), benchSlices(benchPar)
	exps := make([]uint16, benchPar)
	for k := range exps {
		exps[k] = uint16(k)
	}
	b.SetBytes(benchData * benchSlice)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := p.Encode(data, exps, rec); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeCauchy(b *testing.B) {
	e, _ := New(benchData, benchPar)
	shards := benchSlices(benchData + benchPar)
	b.SetBytes(benchData * benchSlice)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := e.Encode(shards); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReconstructPAR2(b *testing.B) {
	p, _ := NewPAR2(benchData)
	data, rec := benchSlices(benchData), benchSlices(benchPar)
	exps := make([]uint16, benchPar)
	for k := range exps {
		exps[k] = uint16(k)
	}
	if err := p.Encode(data, exps, rec); err != nil {
		b.Fatal(err)
	}
	present := make([]bool, benchData)
	for i := range present {
		present[i] = i >= benchPar
	}
	b.SetBytes(benchPar * benchSlice)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := p.Reconstruct(data, present, exps, rec); err != nil {
			b.Fatal(err)
		}
	}
}
