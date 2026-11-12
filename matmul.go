package matmul

import (
	"simd/archsimd"
	"sync"
)

func MatMul(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	res := make([][]int32, len(a))
	for i := range res {
		res[i] = make([]int32, len(b[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[0]); j++ {
			for k := 0; k < len(b[0]); k++ {
				res[i][k] += a[i][j] * b[j][k]
			}
		}
	}

	return res
}

// MatMulTransposed transposes B before computing row dot products.
// Inputs must be rectangular with compatible dimensions, as in MatMul.
func MatMulTransposed(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	transposed := make([][]int32, len(b[0]))
	for k := range transposed {
		transposed[k] = make([]int32, len(b))
		for j := range b {
			transposed[k][j] = b[j][k]
		}
	}

	res := make([][]int32, len(a))
	for i := range a {
		res[i] = make([]int32, len(b[0]))
		for k, row := range transposed {
			var sum int32
			for j := 0; j < len(a[0]); j++ {
				sum += a[i][j] * row[j]
			}
			res[i][k] = sum
		}
	}
	return res
}

// MatMulTransposedSIMD transposes B and uses Go SIMD intrinsics on ARM64
// and requires building with GOEXPERIMENT=simd.
// Inputs must be rectangular with compatible dimensions, as in MatMul.
func MatMulTransposedSIMD(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	transposed := make([][]int32, len(b[0]))
	for k := range transposed {
		transposed[k] = make([]int32, len(b))
		for j := range b {
			transposed[k][j] = b[j][k]
		}
	}
	res := make([][]int32, len(a))
	for i := range a {
		res[i] = make([]int32, len(b[0]))
		for k, row := range transposed {
			var acc archsimd.Int32x4
			j := 0
			for ; j+4 <= len(a[0]); j += 4 {
				av := archsimd.LoadInt32x4(a[i][j:])
				bv := archsimd.LoadInt32x4(row[j:])
				acc = acc.Add(av.Mul(bv))
			}
			sum := acc.ReduceSum()
			for ; j < len(a[0]); j++ {
				sum += a[i][j] * row[j]
			}
			res[i][k] = sum
		}
	}
	return res
}

// MatMulParallel assigns each output row to a separate goroutine.
// Inputs must be rectangular with compatible dimensions, as in MatMul.
func MatMulParallel(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	res := make([][]int32, len(a))
	for i := range res {
		res[i] = make([]int32, len(b[0]))
	}

	var wg sync.WaitGroup
	for i := 0; i < len(a); i++ {
		wg.Go(func() {
			for j := 0; j < len(a[0]); j++ {
				for k := 0; k < len(b[0]); k++ {
					res[i][k] += a[i][j] * b[j][k]
				}
			}
		})
	}
	wg.Wait()
	return res
}

// MatMulParallelTransposed transposes B, then assigns each output row to a goroutine.
// Inputs must be rectangular with compatible dimensions, as in MatMul.
func MatMulParallelTransposed(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	transposed := make([][]int32, len(b[0]))
	for k := range transposed {
		transposed[k] = make([]int32, len(b))
		for j := range b {
			transposed[k][j] = b[j][k]
		}
	}

	res := make([][]int32, len(a))
	for i := range res {
		res[i] = make([]int32, len(b[0]))
	}

	var wg sync.WaitGroup
	for i := range a {
		wg.Go(func() {
			for k, row := range transposed {
				var sum int32
				for j := 0; j < len(a[0]); j++ {
					sum += a[i][j] * row[j]
				}
				res[i][k] = sum
			}
		})
	}
	wg.Wait()
	return res
}

// MatMulParallelTransposedSIMD transposes B, then computes each output row
// in a separate goroutine using SIMD. Requires GOEXPERIMENT=simd on ARM64.
// Inputs must be rectangular with compatible dimensions, as in MatMul.
func MatMulParallelTransposedSIMD(a [][]int32, b [][]int32) [][]int32 {
	if len(a) == 0 || len(b) == 0 {
		panic("empty argument")
	}
	transposed := make([][]int32, len(b[0]))
	for k := range transposed {
		transposed[k] = make([]int32, len(b))
		for j := range b {
			transposed[k][j] = b[j][k]
		}
	}
	res := make([][]int32, len(a))
	for i := range res {
		res[i] = make([]int32, len(b[0]))
	}

	var wg sync.WaitGroup
	for i := range a {
		wg.Go(func() {
			for k, row := range transposed {
				var acc archsimd.Int32x4
				j := 0
				for ; j+4 <= len(a[0]); j += 4 {
					av := archsimd.LoadInt32x4(a[i][j:])
					bv := archsimd.LoadInt32x4(row[j:])
					acc = acc.Add(av.Mul(bv))
				}
				sum := acc.ReduceSum()
				for ; j < len(a[0]); j++ {
					sum += a[i][j] * row[j]
				}
				res[i][k] = sum
			}
		})
	}
	wg.Wait()
	return res
}
