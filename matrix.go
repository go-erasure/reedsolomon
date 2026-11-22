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
	inv, ok := f.tryInvert(m)
	if !ok {
		panic("reedsolomon: singular matrix")
	}
	return inv
}

// pickRows chooses m rows of the n x m matrix c (n >= m) that form an
// invertible square matrix and returns their indices with the inverse of that
// matrix. The first m rows are used when they are invertible; otherwise rows
// are taken greedily, each one kept only if it is independent of those kept
// before it. ok is false when the rows of c have rank below m.
func (f *GF16) pickRows(c matrix, m int) (rows []int, inv matrix, ok bool) {
	// Incremental elimination: basis[col] is a reduced row with a 1 at col. A
	// row is kept only if independent of those kept before it, so the first m
	// rows are chosen whenever they are invertible.
	basis := make([][]uint16, m)
	rows = make([]int, 0, m)
	for r := range c {
		v := append([]uint16(nil), c[r]...)
		pivot := -1
		for col := 0; col < m; col++ {
			if v[col] == 0 {
				continue
			}
			if basis[col] == nil {
				pivot = col
				break
			}
			factor := v[col]
			for j := col; j < m; j++ {
				v[j] ^= f.Mul(factor, basis[col][j])
			}
		}
		if pivot < 0 {
			continue
		}
		scale := v[pivot]
		for j := pivot; j < m; j++ {
			v[j] = f.Div(v[j], scale)
		}
		basis[pivot] = v
		rows = append(rows, r)
		if len(rows) == m {
			break
		}
	}
	// Rank below m leaves zero rows in sub, which tryInvert refuses.
	sub := newMatrix(m, m)
	for k, r := range rows {
		copy(sub[k], c[r])
	}
	inv, ok = f.tryInvert(sub)
	return rows, inv, ok
}

// tryInvert is invert reporting singularity instead of panicking.
func (f *GF16) tryInvert(m matrix) (matrix, bool) {
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
				return nil, false
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
	return inv, true
}
