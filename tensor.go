package main

import (
	"github.com/x448/float16"
	"math"
	"unsafe"
)

type TENSORTYPE int64

const (
	F32  TENSORTYPE = 0
	F16  TENSORTYPE = 1
	BF16 TENSORTYPE = 2
)

type Tensor struct {
	name       string
	shape      []int
	tensortype TENSORTYPE
	data       []float32
	dataq      []byte // quantized data
}

func New2DTensor(n int, m int) *Tensor {
	t := Tensor{}
	t.shape = []int{n, m}
	t.data = make([]float32, n*m)
	return &t
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
	for i := 0; i < t.shape[1]; i++ {
		for j := 0; j < t.shape[0]; j++ {
			newdata[i*t.shape[0]+j] = t.data[j*t.shape[1]+i]
		}
	}
	t.data = newdata
	t.shape[0], t.shape[1] = t.shape[1], t.shape[0]
}

func (t *Tensor) Get2D(i, j int) float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	return t.data[j*t.shape[1]+i]
}

func (t *Tensor) GetRow2D(j int) []float32 {
	if len(t.shape) != 2 {
		panic("Get2D: len(t.shape) != 2")
	}
	offset := j * t.shape[1]
	return t.data[offset : offset+t.shape[1]]
}
