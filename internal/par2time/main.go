// Command par2time times `par2 create` (par2cmdline) on a deterministic file,
// single-threaded, so the PAR2 encoder of this package can be compared with the
// reference tool on the same machine. It reports the median wall time and the
// input throughput (file size / time), which includes par2cmdline's hashing of
// the input: that is the cost of producing a PAR2 set with that tool.
//
//	par2time -par2 ./par2 -dir work -size 6553600 -slice 65536 -recovery 20 -runs 7
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"time"
)

func main() {
	par2 := flag.String("par2", "par2", "par2cmdline binary")
	dir := flag.String("dir", "", "durable work directory (created, emptied of old outputs)")
	size := flag.Int("size", 6553600, "input file size in bytes")
	slice := flag.Int("slice", 65536, "slice (block) size in bytes")
	recovery := flag.Int("recovery", 20, "recovery block count")
	runs := flag.Int("runs", 7, "timed runs")
	flag.Parse()
	if *dir == "" || *runs < 1 {
		fmt.Fprintln(os.Stderr, "par2time: -dir is required")
		os.Exit(2)
	}
	if err := run(*par2, *dir, *size, *slice, *recovery, *runs); err != nil {
		fmt.Fprintln(os.Stderr, "par2time:", err)
		os.Exit(1)
	}
}

func run(par2, dir string, size, slice, recovery, runs int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	buf := make([]byte, size)
	rand.New(rand.NewSource(1)).Read(buf)
	if err := os.WriteFile(filepath.Join(dir, "in.bin"), buf, 0o644); err != nil {
		return err
	}
	var times []time.Duration
	for i := 0; i < runs; i++ {
		old, _ := filepath.Glob(filepath.Join(dir, "*.par2"))
		for _, f := range old {
			os.Remove(f)
		}
		cmd := exec.Command(par2, "create", "-q", "-t1", "-T1",
			fmt.Sprintf("-s%d", slice), fmt.Sprintf("-c%d", recovery), "-n1", "out.par2", "in.bin")
		cmd.Dir = dir
		start := time.Now()
		out, err := cmd.CombinedOutput()
		times = append(times, time.Since(start))
		if err != nil {
			return fmt.Errorf("%v: %s", err, out)
		}
	}
	sort.Slice(times, func(a, b int) bool { return times[a] < times[b] })
	med := times[len(times)/2]
	fmt.Printf("par2 create median %v best %v over %d runs: %.1f MB/s input (median)\n",
		med.Round(time.Millisecond), times[0].Round(time.Millisecond), runs,
		float64(size)/1e6/med.Seconds())
	return nil
}
