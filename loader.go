package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strconv"
)

type BinaryModel struct {
	tensors []*Tensor
	hparams HyperParams
}

var bmodel BinaryModel

func FindTensor(name string) *Tensor {
	for _, t := range bmodel.tensors {
		if t.name == name {
			return t
		}
	}
	panic("Tensor " + name + " not found")
}

func LoadGGML(filename string) {
	buffer := NewReadBufferFromFile(filename)
	magic := buffer.ReadNextInt(4)
	//fmt.Printf("%08x\n", magic)

	if magic != 0x67676a74 {
		panic("Invalid magic " + strconv.FormatInt(int64(magic), 16))
	}
	version := buffer.ReadNextInt(4)
	if version != 3 {
		panic("Invalid version " + strconv.FormatInt(int64(version), 10))
	}
	bmodel.hparams = HyperParams{
		NUMTOKENS: buffer.ReadNextInt(4),
		CTXSIZE:   4096,
		WVSIZE:    buffer.ReadNextInt(4),
		MULT:      buffer.ReadNextInt(4), // Multiple of
		NUMHEADS:  buffer.ReadNextInt(4),
		NUMLAYERS: buffer.ReadNextInt(4),
		N_ROT:     buffer.ReadNextInt(4),
		f16:       buffer.ReadNextInt(4),
		//0 => ggml::TYPE_F32,
		//1 => ggml::TYPE_F16,
		//2 => ggml::TYPE_Q4_0,   https://github.com/ggerganov/ggml/pull/27
		//3 => ggml::TYPE_Q4_1,
		//10 => ggml::TYPE_Q2_K,
	}

	fmt.Printf("%+v\n", bmodel.hparams)
	//n_ff := ((2*(4*bmodel.hparams.WVSIZE)/3 + bmodel.hparams.MULT - 1) / bmodel.hparams.MULT) * bmodel.hparams.MULT
	//fmt.Printf("%d\n", n_ff)

	for i := 0; i < bmodel.hparams.NUMTOKENS; i++ {
		length := buffer.ReadNextInt(4)
		word := buffer.ReadSliceAsString(length)
		//score := buffer.ReadNextFloat32()
		_ = buffer.ReadNextFloat32()
		//fmt.Println(word, score)
		vocab = append(vocab, word)

	}
	for i := 0; ; i++ {
		if buffer.EOF() {
			break
		}
		t := Tensor{}
		nDims := buffer.ReadNextInt(4)
		if buffer.EOF() {
			break
		}
		if nDims > 5 {
			panic("n_dims > 5")
		}
		strLength := buffer.ReadNextInt(4)
		typeId := buffer.ReadNextInt(4)
		size := int64(1)
		for j := 0; j < nDims; j++ {
			dim := buffer.ReadNextInt(4)
			t.shape = append(t.shape, dim)
			size *= int64(dim)
		}
		name := buffer.ReadSliceAsString(strLength)
		t.name = name
		if strLength != 0 {
			bmodel.tensors = append(bmodel.tensors, &t)
		}
		buffer.Align(32)

		var memstats runtime.MemStats
		runtime.ReadMemStats(&memstats)
		fmt.Printf("%-40s | Shape: %v | Type: %2d | Total Memory Usage: %dMB\n", name, t.shape, typeId, memstats.Alloc/1024/1024)
		switch typeId {
		case 0: // f32
			t.data = buffer.ReadNextAsFloat32Array(size)
			t.tensortype = F32
		case 1: // f16
			buffer.SkipNBytes(size * 2)
			t.tensortype = F16
			panic("Not implemented")
			/*
				case 2: // q4_0
					t.dataq4 = buffer.ReadSlice(size / 8 * 5)
					t.tensortype = Q4
					panic("Not implemented")
				case 3: // q4_1
					buffer.SkipNBytes(size / 8 * 6)
					panic("Not implemented")
			*/
		case 10: // Q2_K
			if size%256 != 0 {
				panic("Q2_K size not multiple of 256")
			}
			blocks := size / 256
			// each block bytes: 16 + 64 + 4
			t.dataq = buffer.ReadSlice(blocks * 84)
			t.tensortype = Q2
		case 12: // Q4_K
			if size%256 != 0 {
				panic("Q4_K size not multiple of 256")
			}
			blocks := size / 256
			t.dataq = buffer.ReadSlice(blocks * (4 + 128 + 12))
			t.tensortype = Q4
		case 14: // Q6_K
			if size%256 != 0 {
				panic("Q2_K size not multiple of 256")
			}
			blocks := size / 256
			// each block bytes: 128 + 64 + 16 + 2
			t.dataq = buffer.ReadSlice(blocks * 210)
			t.tensortype = Q6
		default:
			panic("Unknown type " + strconv.FormatInt(int64(typeId), 10))
		}
	}
	fmt.Println("Loaded", len(bmodel.tensors), "tensors")
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
		//fmt.Println(name, safeTensor)
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
			if safeTensor.DataOffsets[1]-safeTensor.DataOffsets[0] != size {
				panic("Error: size mismatch")
			}
			t.tensortype = F32
		case "F16":
			size *= 2
			//fmt.Println(safeTensor.DataOffsets[1]-safeTensor.DataOffsets[0], size)
			if safeTensor.DataOffsets[1]-safeTensor.DataOffsets[0] != size {
				panic("Error: size mismatch")
			}
			t.tensortype = F16
		default:
			panic("Unknown type " + safeTensor.Dtype)
		}
		buffer.Seek(int64(safeTensor.DataOffsets[0]))
		t.dataq = buffer.ReadSlice(size)
		//t.ToFloat32()
	}
}
