package reedsolomon

import "errors"

// This file implements the recovery coding of the PAR 2.0 format, so that
// recovery slices produced here are byte-identical to those of par2cmdline and
// can be consumed by it, and the other way round.
//
// Sources: the PAR 2.0 specification (https://parchive.github.io/doc/Parity_Volume_Set_Specification_v2.0.html,
// "Main Packet" and "Recovery Slice Packet": 16-bit Galois field with generator
// 0x1100B, "All integers are Intel-endian", one constant per input slice, the
// recovery slice of exponent e is the sum over the input slices i of
// constant_i^e * slice_i) and par2cmdline v2.0.0 src/reedsolomon.cpp:
//
//   - SetInput, lines 232-263: the constant of input slice i is 2^logbase where
//     logbase is the i-th non-negative integer (starting at 0) that is coprime to
//     65535 (the loop at 249 skips every logbase with gcd(65535, logbase) != 1).
//   - InternalProcess, lines 343-355: slices are arrays of 16-bit words cast
//     from the byte buffer ("BUG: This only works for __LITTLE_ENDIAN"), i.e.
//     little-endian words on every host that produced real-world PAR2 files.
//   - Compute / reedsolomon.h:214-218: the matrix entry of input i and output
//     (recovery) block e is constant_i^exponent_e.

// ErrSingular is returned by PAR2.Reconstruct when the recovery slices given do
// not determine the missing data slices. PAR 2.0's Vandermonde construction is
// not guaranteed to be invertible for every choice of exponents (the
// specification acknowledges this); supplying further recovery slices may help.
var ErrSingular = errors.New("reedsolomon: recovery slices do not determine the missing data")

// MaxPAR2DataSlices is the largest number of input slices PAR 2.0 can address:
// the constants are the powers 2^k with k coprime to 65535, and there are
// phi(65535) = 32768 of them.
const MaxPAR2DataSlices = 32768

// PAR2 is a PAR 2.0 recovery codec for a fixed number of data slices. Slices
// are byte slices of equal, even length holding little-endian 16-bit words, and
// each recovery slice carries a 16-bit exponent (the Recovery Slice packet's
// exponent field). It is a separate type from Encoder on purpose: Encoder is a
// systematic Cauchy code over big-endian words, PAR2 is the Vandermonde code
// over little-endian words, and the two are not interchangeable.
type PAR2 struct {
	field *GF16
	// logs[i] is the discrete logarithm of the constant of data slice i, so
	// constant_i^e = 2^(logs[i]*e).
	logs []uint16
}

// NewPAR2 returns the PAR 2.0 codec for dataSlices input slices, 1 to
// MaxPAR2DataSlices.
func NewPAR2(dataSlices int) (*PAR2, error) {
	if dataSlices <= 0 || dataSlices > MaxPAR2DataSlices {
		return nil, ErrInvalidParams
	}
	logs := make([]uint16, dataSlices)
	logbase := 0
	for i := range logs {
		for gcd(gf16Nonzero, logbase) != 1 {
			logbase++
		}
		logs[i] = uint16(logbase)
		logbase++
	}
	return &PAR2{field: NewGF16(), logs: logs}, nil
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// DataSlices returns the number of data slices.
func (p *PAR2) DataSlices() int { return len(p.logs) }

// Constant returns the PAR 2.0 constant of data slice i (the "base" of the
// Vandermonde matrix column).
func (p *PAR2) Constant(i int) uint16 { return p.field.Exp(int(p.logs[i])) }

// Factor returns the coefficient of data slice i in the recovery slice of the
// given exponent: Constant(i)^exponent.
func (p *PAR2) Factor(i int, exponent uint16) uint16 {
	return p.field.Exp(int(p.logs[i]) * int(exponent))
}

func (p *PAR2) checkData(data [][]byte) error {
	if len(data) != len(p.logs) {
		return ErrShardCount
	}
	return nil
}

// checkSizes verifies that data and recovery slices all share one even length.
func checkSizes(data, recovery [][]byte) error {
	all := make([][]byte, 0, len(data)+len(recovery))
	all = append(all, data...)
	all = append(all, recovery...)
	_, err := checkShards(all)
	return err
}

// Encode computes the recovery slices of the given exponents from the data
// slices: recovery[k] = sum over i of Constant(i)^exponents[k] * data[i]. The
// recovery slices are overwritten and must be allocated to the data slice
// length.
func (p *PAR2) Encode(data [][]byte, exponents []uint16, recovery [][]byte) error {
	if err := p.checkData(data); err != nil {
		return err
	}
	if len(exponents) != len(recovery) {
		return ErrShardCount
	}
	if err := checkSizes(data, recovery); err != nil {
		return err
	}
	for k, e := range exponents {
		p.combine(recovery[k], data, e)
	}
	return nil
}

// combine sets dst = sum over i of Factor(i, e) * src[i].
func (p *PAR2) combine(dst []byte, src [][]byte, e uint16) {
	galMulLE(p.field, dst, src[0], p.Factor(0, e))
	for i := 1; i < len(src); i++ {
		galMulAddLE(p.field, dst, src[i], p.Factor(i, e))
	}
}

// Verify reports whether the recovery slices are consistent with the data
// slices.
func (p *PAR2) Verify(data [][]byte, exponents []uint16, recovery [][]byte) (bool, error) {
	if err := p.checkData(data); err != nil {
		return false, err
	}
	if len(exponents) != len(recovery) {
		return false, ErrShardCount
	}
	if err := checkSizes(data, recovery); err != nil {
		return false, err
	}
	scratch := make([]byte, len(data[0]))
	for k, e := range exponents {
		p.combine(scratch, data, e)
		if string(scratch) != string(recovery[k]) {
			return false, nil
		}
	}
	return true, nil
}

// Reconstruct recovers the missing data slices in place. present[i] == false
// marks data slice i as lost; its slice must already be allocated to the common
// length. recovery holds the available recovery slices and exponents their
// exponents. It needs as many independent recovery slices as there are missing
// data slices: ErrTooFewShards if there are fewer, ErrSingular if the ones
// given are dependent. Surplus recovery slices stand in for any that turn out
// to be dependent, as in par2cmdline. recovery is not modified.
func (p *PAR2) Reconstruct(data [][]byte, present []bool, exponents []uint16, recovery [][]byte) error {
	if err := p.checkData(data); err != nil {
		return err
	}
	if len(present) != len(data) || len(exponents) != len(recovery) {
		return ErrShardCount
	}
	var missing []int
	for i, ok := range present {
		if !ok {
			missing = append(missing, i)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	if err := checkSizes(data, recovery); err != nil {
		return err
	}
	if len(recovery) < len(missing) {
		return ErrTooFewShards
	}

	m := len(missing)
	// coeff[r][c] = Factor(missing[c], exponents[r]) for every available row.
	coeff := newMatrix(len(recovery), m)
	for r, e := range exponents {
		for c, i := range missing {
			coeff[r][c] = p.Factor(i, e)
		}
	}
	rows, inv, ok := p.field.pickRows(coeff, m)
	if !ok {
		return ErrSingular
	}

	// rhs[k] = recovery[rows[k]] minus the contribution of the present slices.
	rhs := make([][]byte, m)
	for k, r := range rows {
		rhs[k] = append([]byte(nil), recovery[r]...)
		for i := range data {
			if present[i] {
				galMulAddLE(p.field, rhs[k], data[i], p.Factor(i, exponents[r]))
			}
		}
	}
	// data[missing[c]] = sum over k of inv[c][k] * rhs[k].
	for c, i := range missing {
		galMulLE(p.field, data[i], rhs[0], inv[c][0])
		for k := 1; k < m; k++ {
			galMulAddLE(p.field, data[i], rhs[k], inv[c][k])
		}
	}
	return nil
}
