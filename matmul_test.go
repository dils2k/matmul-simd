package matmul

import (
	"math/rand"
	"reflect"
	"testing"
)

func BenchmarkMatMul200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMul)
}

func BenchmarkMatMulTransposed200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMulTransposed)
}

func BenchmarkMatMulTransposedSIMD200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMulTransposedSIMD)
}

func BenchmarkMatMulParallel200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMulParallel)
}

func BenchmarkMatMulParallelTransposed200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMulParallelTransposed)
}

func BenchmarkMatMulParallelTransposedSIMD200x200(b *testing.B) {
	benchmarkMatMul200x200(b, MatMulParallelTransposedSIMD)
}

func benchmarkMatMul200x200(b *testing.B, multiply func([][]int32, [][]int32) [][]int32) {
	const size = 200
	a := make([][]int32, size)
	right := make([][]int32, size)
	for i := range a {
		a[i] = make([]int32, size)
		right[i] = make([]int32, size)
		for j := range a[i] {
			a[i][j] = int32((i+j)%10 + 1)
			right[i][j] = int32((i*3+j)%10 + 1)
		}
	}

	b.ReportAllocs()
	for b.Loop() {
		multiply(a, right)
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "ops/s")
}

func TestMatMulImplementations(t *testing.T) {
	for _, shape := range [][3]int{{1, 1, 1}, {2, 3, 4}, {5, 2, 3}, {100, 100, 100}} {
		m, n, p := shape[0], shape[1], shape[2]
		a, b := make([][]int32, m), make([][]int32, n)
		for i := range a {
			a[i] = make([]int32, n)
			for j := range a[i] {
				a[i][j] = int32(i - j)
			}
		}
		for i := range b {
			b[i] = make([]int32, p)
			for j := range b[i] {
				b[i][j] = int32(i + j + 1)
			}
		}
		want := MatMul(a, b)
		for _, impl := range []struct {
			name     string
			multiply func([][]int32, [][]int32) [][]int32
		}{
			{"transposed", MatMulTransposed},
			{"transposed_simd", MatMulTransposedSIMD},
			{"parallel", MatMulParallel},
			{"parallel_transposed", MatMulParallelTransposed},
			{"parallel_transposed_simd", MatMulParallelTransposedSIMD},
		} {
			if got := impl.multiply(a, b); !reflect.DeepEqual(got, want) {
				t.Fatalf("%s shape %v: got %v, want %v", impl.name, shape, got, want)
			}
		}
	}
}

func TestMatMulTransposedSIMDFullWidth(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for n := 1; n <= 257; n++ {
		a, b := [][]int32{make([]int32, n)}, make([][]int32, n)
		for i := range a[0] {
			a[0][i] = int32(rng.Uint32())
			b[i] = []int32{int32(rng.Uint32())}
		}
		want := MatMul(a, b)
		for _, multiply := range []func([][]int32, [][]int32) [][]int32{
			MatMulTransposedSIMD, MatMulParallelTransposedSIMD,
		} {
			if got := multiply(a, b); !reflect.DeepEqual(got, want) {
				t.Fatalf("length %d: got %v, want %v", n, got, want)
			}
		}
	}
}
