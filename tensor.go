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
	dataq      []byte // quantized data
}

func (t *Tensor) bytesToFloat32() {
	size := t.GetSize()
	t.data = make([]float32, size)

	data := unsafe.Slice((*uint32)(unsafe.Pointer(&t.dataq[0])), size)
	for i := int64(0); i < size; i++ {
		//bits := binary.LittleEndian.Uint32(t.dataq[i*4:])
		t.data[i] = math.Float32frombits(data[i])
	}
	t.dataq = nil
	t.tensortype = F32
}

func (t *Tensor) bytesFromFloat16ToFloat32() {
	size := t.GetSize()
	newdata := make([]float32, size)

	data := unsafe.Slice((*uint16)(unsafe.Pointer(&t.dataq[0])), size)
	for i := int64(0); i < size; i++ {
		//newdata[i] = float16.Frombits(uint16(t.dataq[i*2+0]) | uint16(t.dataq[i*2+1])<<8).Float32()
		newdata[i] = float16.Frombits(data[i]).Float32()
	}
	t.dataq = nil
	t.data = newdata
	t.tensortype = F32
}

func (t *Tensor) ToFloat32() {
	switch t.tensortype {
	case F32:
		t.bytesToFloat32()
	case F16:
		t.bytesFromFloat16ToFloat32()
	default:
		panic("tensortype unknown")
	}
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
	return t.data[j*t.shape[1]+i]
}

func (t *Tensor) GetRow2D(j int) []float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	offset := j * t.shape[1]
	return t.data[offset : offset+t.shape[1]]
}

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
