package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// run must refuse, with an error, a work directory it cannot create, one it
// cannot write the input to, and a par2 binary that fails.
func TestRunRefusals(t *testing.T) {
	dir := t.TempDir()

	file := filepath.Join(dir, "plain-file")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run("par2", filepath.Join(file, "sub"), 64, 64, 1, 1); err == nil || !strings.Contains(err.Error(), "mkdir") {
		t.Fatalf("work directory under a regular file: %v", err)
	}

	ro := filepath.Join(dir, "ro")
	if err := os.Mkdir(ro, 0o555); err != nil {
		t.Fatal(err)
	}
	if os.Getuid() != 0 {
		if err := run("par2", ro, 64, 64, 1, 1); err == nil || !strings.Contains(err.Error(), "in.bin") {
			t.Fatalf("unwritable work directory: %v", err)
		}
	}

	err := run(filepath.Join(dir, "no-such-par2"), filepath.Join(dir, "work"), 64, 64, 1, 1)
	if err == nil {
		t.Fatal("missing par2 binary accepted")
	}
	if !strings.Contains(err.Error(), "no-such-par2") {
		t.Fatalf("error does not name the binary: %v", err)
	}
}
