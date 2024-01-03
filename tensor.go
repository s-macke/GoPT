package main

import (
	"encoding/binary"
	"github.com/x448/float16"
	"math"
	"unsafe"
)

type TENSORTYPE int64

const (
	F32 TENSORTYPE = 0
	F16 TENSORTYPE = 1
	Q4  TENSORTYPE = 2
	Q2  TENSORTYPE = 3
	Q6  TENSORTYPE = 4
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

func (t *Tensor) Dequantize() {
	if t.tensortype != Q4 {
		panic("Dequantize: t.tensortype != Q4")
	}
	if len(t.shape) == 2 {
		t.Dequantize2D()
	} else {
		panic("Dequantize for 1D not implemented")
		//t.Dequantize1D()
	}
}

func (t *Tensor) Dequantize2D() {
	t.data = make([]float32, t.shape[0]*t.shape[1])
	n := t.shape[0] * 5 / 8
	for j := 0; j < t.shape[1]; j++ {
		t.Dequantize1D(t.dataq[j*n:(j+1)*n], t.data[j*t.shape[0]:(j+1)*t.shape[0]])
		//t.data[i*t.shape[1]+j] = float32(t.dataq4[i*t.shape[1]+j]) / 16
	}
}

func (t *Tensor) Dequantize1D(input []byte, output []float32) {
	//fmt.Println(len(input), len(output))
	const QK = 32

	nb := len(output) / QK // number of blocks
	bs := QK/2 + 4         // block size

	pd := input[0:]
	pb := input

	for i := 0; i < nb; i++ {
		d := math.Float32frombits(binary.LittleEndian.Uint32(pd[i*bs:]))
		pp := pb[i*bs+4:]

		for j := 0; j < QK/2; j++ {
			vi0 := pp[j] & 0xF
			vi1 := (pp[j] >> 4) & 0xF
			output[i*QK+j*2+0] = (float32(vi0) - 8.) * d
			output[i*QK+j*2+1] = (float32(vi1) - 8.) * d
		}
		//fmt.Println(d)
	}

	// dequantize row q4_0 k=4096 QK=32 sizeof(float)=4 bs=20

	//panic("Dequantize1D not implemented")
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
