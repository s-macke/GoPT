package main

import (
	"fmt"
	"runtime"
)

var tensors []*tensor

func FindTensor(name string) *tensor {
	for _, t := range tensors {
		if t.name == name {
			return t
		}
	}
	panic("Tensor " + name + " not found")
}

func LoadBin(filename string) {
	buffer := NewReadBufferFromFile2(filename)
	for i := 0; ; i++ {
		//fmt.Println("Memory:", memstats.Alloc/1024/1024, "MB")
		if buffer.EOF() {
			break
		}
		//fmt.Println("-----", i, "-----")
		buffer.SkipNBytes(4)
		t := tensor{}
		tensors = append(tensors, &t)
		type_id := buffer.ReadNextInt(4)
		if type_id != 0 {
			panic("type_id != 0, Only float32 supported")
		}
		n_dims := buffer.ReadNextInt(4)
		str_length := buffer.ReadNextInt(4)
		//fmt.Println("type:", type_id)
		//fmt.Println("shape:", n_dims)
		//fmt.Println(str_length)
		size := 1
		for i := 0; i < n_dims; i++ {
			dim := buffer.ReadNextInt(4)
			t.shape = append(t.shape, dim)
			size *= dim
			//fmt.Print(dim, " ")
		}
		//fmt.Println()
		name := buffer.ReadSliceAsString(str_length)
		t.name = name
		//fmt.Println(name)
		//fmt.Println(size, "parameters")
		var memstats runtime.MemStats
		runtime.ReadMemStats(&memstats)
		fmt.Println("Loading", name, t.shape, "type", type_id, "Memory:", memstats.Alloc/1024/1024, "MB")
		switch type_id {
		case 0:
			t.data = buffer.ReadNextAsFloat32Array(size)
		case 2:
			buffer.SkipNBytes(size * 2)
		default:
			panic("Unknown type")
		}
	}
	fmt.Println("Loaded", len(tensors), "tensors")
}
