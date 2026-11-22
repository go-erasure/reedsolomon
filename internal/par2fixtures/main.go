// Command par2fixtures writes the deterministic input files of the PAR2
// interoperability fixtures in testdata/par2. The .par2 files next to them were
// produced from these inputs by par2cmdline; see testdata/par2/README.md for
// the exact commands.
//
//	go run ./internal/par2fixtures testdata/par2
package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
)

// inputs are the fixture inputs: name and length. The contents are the output
// of math/rand seeded with the length, so they never change.
var inputs = []struct {
	name string
	size int
}{
	{"a.bin", 1000},
	{"b.bin", 300},
	{"c.bin", 777},
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: par2fixtures <dir>")
		os.Exit(2)
	}
	for _, in := range inputs {
		buf := make([]byte, in.size)
		rand.New(rand.NewSource(int64(in.size))).Read(buf)
		if err := os.WriteFile(filepath.Join(os.Args[1], in.name), buf, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
