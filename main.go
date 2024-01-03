package main

import (
	"fmt"
	"os"
	"runtime"
)

func main() {
	// Check https://github.com/karpathy/llama2.c/blob/master/run.c
	// https://github.com/mukel/llama2.java
	// for the original code

	//LoadBin("ggml-alpaca-7b-q4.bin")
	//LoadGGML("llama-2-7b-chat.ggmlv3.q2_K.bin")
	//LoadSafetensors("../llama2/gptq/gptq_model-4bit-128g.safetensors")
	LoadSafetensors("mamba.safetensors")

	runtime.GC()
	for _, t := range bmodel.tensors {
		t.ToFloat32()
	}

	os.Exit(1)

	m := NewModel(bmodel.hparams)

	tokens := Translate("Building a website can be done in 10 simple steps:")
	fmt.Println(tokens)

	// some experiments with word vectors
	similarity(m)
	relation(m)
	/*
		tokens := Translate(" Suddenly, a magical floppy disk")
		m.SetTemperature(1.2)
		m.SetTokens(tokens)
		m.Run()
	*/
}
