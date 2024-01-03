package main

import (
	"runtime"
	"testing"
)

func TestFindTensor(t *testing.T) {
	LoadSafetensors("model-00001-of-00002.safetensors")
	LoadSafetensors("model-00002-of-00002.safetensors")

	runtime.GC()
	for _, t := range bmodel.tensors {
		t.ToFloat32()
	}

}
