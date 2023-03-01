package main

import (
	_ "embed"
	"math"
	"os"
	"reflect"
	"unsafe"
)

type ReadBuffer struct {
	b      []byte
	offset int
}

func NewReadBuffer(b []byte) *ReadBuffer {
	return &ReadBuffer{
		b:      b,
		offset: 0,
	}
}

func NewReadBufferFromFile(filename string) *ReadBuffer {
	b, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	return &ReadBuffer{
		b:      b,
		offset: 0,
	}
}

func (rb *ReadBuffer) Length() int {
	return len(rb.b)
}

func (rb *ReadBuffer) ReadNextByte() byte {
	if rb.offset >= len(rb.b) {
		return 0
	}
	b := rb.b[rb.offset]
	rb.offset++
	return b
}

func (rb *ReadBuffer) ReadSlice(n int) []byte {
	//fmt.Println("ReadSlice", n, rb.offset, len(rb.b))
	b := rb.b[rb.offset : rb.offset+n]
	rb.offset = rb.offset + n
	return b
}

func (rb *ReadBuffer) ReadSliceAsString(n int) string {
	bytes := rb.ReadSlice(n)
	for i := 0; i < len(bytes); i++ {
		if bytes[i] == 0 {
			return string(bytes[:i])
		}
	}
	return string(bytes)
}

func clen(n []byte) int {
	for i := 0; i < len(n); i++ {
		if n[i] == 0 {
			return i
		}
	}
	return len(n)
}
func (rb *ReadBuffer) ReadSliceAsNullTerminatedString(n int) string {
	bytes := rb.ReadSlice(n)
	length := clen(bytes)
	return string(bytes[:length])
}

func (rb *ReadBuffer) NewReadBuffer(n int) *ReadBuffer {
	b := rb.b[rb.offset : rb.offset+n]
	rb.offset = rb.offset + n
	return NewReadBuffer(b)
}

func (rb *ReadBuffer) NewReadBufferAt(offset int) *ReadBuffer {
	b := rb.b[offset:]
	return NewReadBuffer(b)
}

func (rb *ReadBuffer) SkipNBytes(n int) {
	rb.offset = rb.offset + n
}

func (rb *ReadBuffer) EOF() bool {
	return rb.offset >= len(rb.b)
}

func (rb *ReadBuffer) ReadNextInt(bytes int) int {
	var value = 0
	for i := 0; i < bytes; i++ {
		value |= int(rb.ReadNextByte()) << (8 * i)
	}
	return value
}

func (rb *ReadBuffer) ReadNextFloat32() float32 {
	return math.Float32frombits(uint32(rb.ReadNextInt(4)))
}

func (rb *ReadBuffer) ReadNextAsFloat32Array(n int) []float32 {
	data := *(*([]float32))(unsafe.Pointer(&rb.b))

	sh := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sh.Len = n
	sh.Cap = n
	sh.Data += uintptr(rb.offset)

	/*
		sh = (*reflect.SliceHeader)(unsafe.Pointer(data))
		fmt.Printf("%+v\n", sh)
	*/
	rb.offset += n * 4
	return data
}
