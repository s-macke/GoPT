package main

import (
	"bufio"
	_ "embed"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
)

type ReadBuffer struct {
	offset int
	file   *os.File
	reader *bufio.Reader
	eof    bool
}

func (rb *ReadBuffer) checkError(err error) {
	if err == nil {
		return
	}
	if errors.Is(err, io.EOF) {
		rb.eof = true
		return
	}
	panic(err)
}

func NewReadBufferFromFile2(filename string) *ReadBuffer {
	file, err := os.Open(filename)

	if err != nil {
		panic(err)
	}
	return &ReadBuffer{
		file:   file,
		eof:    false,
		offset: 0,
	}
}

func (rb *ReadBuffer) EOF() bool {
	return rb.eof
}

func (rb *ReadBuffer) SkipNBytes(n int) {

	_, err := rb.file.Seek(int64(n), io.SeekCurrent)
	rb.checkError(err)
	rb.offset = rb.offset + n
}

func (rb *ReadBuffer) ReadNextByte() byte {
	var bytes [1]byte
	_, err := rb.file.Read(bytes[:])
	rb.checkError(err)
	rb.offset++
	return bytes[0]
}

func (rb *ReadBuffer) ReadNextInt(bytes int) int {
	var value = 0
	for i := 0; i < bytes; i++ {
		value |= int(rb.ReadNextByte()) << (8 * i)
	}
	return value
}

func (rb *ReadBuffer) Length() int {
	info, err := rb.file.Stat()
	rb.checkError(err)
	return int(info.Size())
}

func (rb *ReadBuffer) ReadSlice(n int) []byte {
	b := make([]byte, n)
	_, err := rb.file.Read(b)
	rb.checkError(err)
	//fmt.Println("ReadSlice", n, rb.offset, len(rb.b))
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

func clen2(n []byte) int {
	for i := 0; i < len(n); i++ {
		if n[i] == 0 {
			return i
		}
	}
	return len(n)
}
func (rb *ReadBuffer) ReadSliceAsNullTerminatedString(n int) string {
	bytes := rb.ReadSlice(n)
	length := clen2(bytes)
	return string(bytes[:length])
}

func (rb *ReadBuffer) ReadNextFloat32() float32 {
	return math.Float32frombits(uint32(rb.ReadNextInt(4)))
}

func (rb *ReadBuffer) ReadNextAsFloat32Array(n int) []float32 {
	floats := make([]float32, n)
	err := binary.Read(rb.file, binary.LittleEndian, &floats)
	rb.checkError(err)
	rb.offset += n * 4
	return floats
}
