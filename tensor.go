package main

import (
	"github.com/x448/float16"
	"math"
	"unsafe"
)

type TENSORTYPE int64

const (
	F32 TENSORTYPE = 0
	F16 TENSORTYPE = 1
)

type Tensor struct {
	name       string
	shape      []int
	tensortype TENSORTYPE
	data       []float32
	dataq      []byte
}

func (t *Tensor) ToFloat32() {
	if t.tensortype == F32 {
		return
	}
	if t.tensortype != F16 {
		panic("ToFloat32: t.tensortype != F16")
	}
	size := t.GetSize()
	//fmt.Println(size)
	newdata := make([]float32, size)

	data := unsafe.Slice((*uint16)(unsafe.Pointer(&t.dataq[0])), size)
	for i := int64(0); i < size; i++ {
		//newdata[i] = float16.Frombits(uint16(t.dataq[i*2+0]) | uint16(t.dataq[i*2+1])<<8).Float32()
		newdata[i] = float16.Frombits(data[i]).Float32()
		//newdata[i] = float32(data[i])
		//newdata[i] = 0.
	}
	t.dataq = nil
	t.data = newdata
	t.tensortype = F32
}

func (t *Tensor) GetSize() int64 {
	size := int64(1)
	for _, v := range t.shape {
		size *= int64(v)
	}
	return size
}

func (t *Tensor) Transpose() {
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

func (t *Tensor) Get2D(i, j int) float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	//return t.data[i*t.shape[1]+j]
	return t.data[j*t.shape[0]+i]
}

func (t *Tensor) GetRow2D(j int) []float32 {
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
