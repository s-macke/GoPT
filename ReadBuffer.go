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
	offset int64
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

func NewReadBufferFromFile(filename string) *ReadBuffer {
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

func (rb *ReadBuffer) Seek(offset int64) {
	var err error
	rb.offset, err = rb.file.Seek(offset, io.SeekStart)
	rb.checkError(err)
}

func (rb *ReadBuffer) SkipNBytes(n int64) {
	_, err := rb.file.Seek(n, io.SeekCurrent)
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

func (rb *ReadBuffer) ReadSlice(n int64) []byte {
	b := make([]byte, n)
	readBytes, err := rb.file.Read(b)
	rb.checkError(err)
	if readBytes != int(n) {
		panic("ReadSlice: readBytes != n")
	}
	//fmt.Println("ReadSlice", n, rb.offset, len(rb.b))
	rb.offset = rb.offset + n
	return b
}

func (rb *ReadBuffer) Align(n int) {
	//fmt.Println("Align", n, rb.offset, len(rb.b))
	if rb.offset%int64(n) != 0 {
		rb.SkipNBytes(int64(n) - (rb.offset % int64(n)))
	}
}

func (rb *ReadBuffer) ReadSliceAsString(n int) string {
	bytes := rb.ReadSlice(int64(n))
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
	bytes := rb.ReadSlice(int64(n))
	length := clen2(bytes)
	return string(bytes[:length])
}

func (rb *ReadBuffer) ReadNextFloat32() float32 {
	return math.Float32frombits(uint32(rb.ReadNextInt(4)))
}

func (rb *ReadBuffer) ReadNextAsFloat32Array(n int64) []float32 {
	floats := make([]float32, n)
	err := binary.Read(rb.file, binary.LittleEndian, &floats)
	rb.checkError(err)
	rb.offset += int64(n) * 4
	return floats
}
