package main

import (
	"math"
)

type tensor struct {
	name  string
	shape []int
	data  []float32
}

func (t *tensor) Transpose() {
	if len(t.shape) != 2 {
		panic("Transpose: len(t.shape) != 2")
	}
	if t.shape[0]*t.shape[1] != len(t.data) {
		panic("Transpose: t.shape[0]*t.shape[1] != len(t.data)")
	}

	newdata := make([]float32, len(t.data))
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			newdata[i*t.shape[1]+j] = t.data[j*t.shape[0]+i]
		}
	}
	t.data = newdata

	t.shape[0], t.shape[1] = t.shape[1], t.shape[0]
}

func (t *tensor) Get2D(i, j int) float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	//return t.data[i*t.shape[1]+j]
	return t.data[j*t.shape[0]+i]
}

func (t *tensor) GetRow2D(j int) []float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	offset := j * t.shape[0]
	return t.data[offset : offset+t.shape[0]]
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

func scalarProduct(v []float32, m []float32) float32 {
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
