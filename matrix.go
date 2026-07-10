package reedsolomon

// matrix is a dense matrix of GF(2^16) elements stored row-major.
type matrix [][]uint16

// newMatrix allocates a rows x cols matrix of zeros.
func newMatrix(rows, cols int) matrix {
	m := make(matrix, rows)
	for i := range m {
		m[i] = make([]uint16, cols)
	}
	return m
}

// invert returns the inverse of the square matrix m over GF(2^16) using
// Gauss-Jordan elimination on m augmented with the identity. It panics if m is
// singular; callers in this package only ever pass sub-matrices of an MDS
// generator matrix, which are always invertible.
func (f *GF16) invert(m matrix) matrix {
	n := len(m)
	work := newMatrix(n, 2*n)
	for i := 0; i < n; i++ {
		copy(work[i], m[i])
		work[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		if work[col][col] == 0 {
			found := -1
			for r := col + 1; r < n; r++ {
				if work[r][col] != 0 {
					found = r
					break
				}
			}
			if found < 0 {
				panic("reedsolomon: singular matrix")
			}
			work[col], work[found] = work[found], work[col]
		}
		pivot := work[col][col]
		for j := 0; j < 2*n; j++ {
			work[col][j] = f.Div(work[col][j], pivot)
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := work[r][col]
			if factor == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				work[r][j] ^= f.Mul(factor, work[col][j])
			}
		}
	}
	inv := newMatrix(n, n)
	for i := 0; i < n; i++ {
		copy(inv[i], work[i][n:])
	}
	return inv
}
