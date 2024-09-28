package main

import "math"

// Root Mean Square Layer Normalization together with weight multiplication
// https://en.wikipedia.org/wiki/Root_mean_square
func RMSNorm(x []float32, weight []float32) {
	n := len(x)
	if n != len(weight) {
		panic("RMSNorm: len(x) != len(weight)")
	}
	const eps = 1e-5

	var sum float32 = 0
	for i := 0; i < n; i++ {
		sum += x[i] * x[i]
	}

	rsqrt := 1. / float32(math.Sqrt(float64(sum/float32(n)+eps)))
	for i := 0; i < n; i++ {
		x[i] *= weight[i] * rsqrt
	}
}

const EPSILON = 0.0000001

// Normalize normalizes the input vector x and stores the result in o.
func Normalize(o, x, b, g []float32) {
	var mean float32 = 0.
	var smean float32 = 0.
	var muller float32 = 0.

	SIZE := len(x)

	for i := 0; i < SIZE; i++ {
		mean += x[i]
	}
	mean /= float32(SIZE)
	for i := 0; i < SIZE; i++ {
		a := x[i] - mean
		smean += a * a
	}
	smean /= float32(SIZE)
	if smean < EPSILON {
		smean = EPSILON
	}

	muller = float32(math.Sqrt(1. / float64(smean)))
	if b != nil {
		for i := 0; i < SIZE; i++ {
			o[i] = (x[i]-mean)*muller*g[i] + b[i]
		}
	} else {
		for i := 0; i < SIZE; i++ {
			o[i] = (x[i] - mean) * muller * g[i]
		}
	}
}

func ScalarProduct(v []float32, m []float32) float32 {
	if len(v) != len(m) {
		panic("scalarProduct: len(v) != len(m)")
	}
	l := len(v)
	var a float32 = 0
	for i := 0; i < l; i++ {
		a += v[i] * m[i]
	}
	return a
}
