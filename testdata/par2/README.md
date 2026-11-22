# PAR2 fixtures

Real output of the reference tool, used as the oracle by `par2_test.go`
(`TestPAR2MatchesPar2cmdline`, `TestPAR2ReconstructsFromPar2cmdline`).

Reference: par2cmdline **2.0.0** (tag `v2.0.0` of
https://github.com/Parchive/par2cmdline), built with CMake, Release. Not
par2cmdline-turbo.

The input files are deterministic; regenerate them from the repository root with

    go run ./internal/par2fixtures testdata/par2/single
    go run ./internal/par2fixtures testdata/par2/multi

and the recovery sets with

    cd testdata/par2/single && par2 create -q -s128 -c6 -n1 -f0 a.par2 a.bin
    cd testdata/par2/multi  && par2 create -q -s64  -c5 -n1 -f3 m.par2 b.bin c.bin

`single`: one 1000-byte file, 128-byte slices, 6 recovery slices (exponents
0..5). `multi`: two files (300 and 777 bytes), 64-byte slices, 5 recovery slices
with first exponent 3 (`-f3`), which exercises a non-zero base exponent and
slices that span two files.
