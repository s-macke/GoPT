package main

import (
	"encoding/json"
	"fmt"
	"runtime"
)

type BinaryModel struct {
	tensors []*Tensor
}

var bmodel BinaryModel

func DoesExistTensorByName(name string) bool {
	for _, t := range bmodel.tensors {
		if t.name == name {
			return true
		}
	}
	return false
}

func GetTensorByName(name string) *Tensor {
	for _, t := range bmodel.tensors {
		if t.name == name {
			return t
		}
	}
	panic("Tensor " + name + " not found")
}

type SafeTensor struct {
	Dtype       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func LoadSafetensors(filename string) {
	buffer := NewReadBufferFromFile(filename)
	headerSize := buffer.ReadNextInt(8)
	str := buffer.ReadSliceAsString(headerSize)

	var mapTensors map[string]SafeTensor
	err := json.Unmarshal([]byte(str), &mapTensors)
	if err != nil {
		panic(err)
	}

	for name, safeTensor := range mapTensors {
		if name == "__metadata__" {
			continue
		}
		var memstats runtime.MemStats
		runtime.ReadMemStats(&memstats)
		fmt.Printf("%-50s | Type: %5s | Shape: %v | Total Memory Usage: %dMB\n", name, safeTensor.Dtype, safeTensor.Shape, memstats.Alloc/1024/1024)

		t := Tensor{}
		bmodel.tensors = append(bmodel.tensors, &t)

		size := int64(1)
		for _, dim := range safeTensor.Shape {
			size *= int64(dim)
		}
		t.shape = safeTensor.Shape
		t.name = name

		switch safeTensor.Dtype {
		case "F32":
			size *= 4
			t.tensortype = F32
		case "F16":
			size *= 2
			//fmt.Println(safeTensor.DataOffsets[1]-safeTensor.DataOffsets[0], size)
			t.tensortype = F16
		default:
			panic("Unknown type " + safeTensor.Dtype)
		}
		if safeTensor.DataOffsets[1]-safeTensor.DataOffsets[0] != size {
			panic("Error: size mismatch")
		}
		buffer.Seek(int64(headerSize+8) + safeTensor.DataOffsets[0])
		t.dataq = buffer.ReadSlice(size)
		//t.ToFloat32()
	}
}
