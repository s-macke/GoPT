package main

import (
	"fmt"
	"runtime"
)

func main() {
	LoadSafetensors("mamba.safetensors")
	vocab := NewVocabulary("mamba_vocab.json")
	fmt.Println("Vocabulary size:", len(vocab.vocab))

	runtime.GC()
	for _, t := range bmodel.tensors {
		t.ToFloat32()
	}
	runtime.GC()

	m := NewModel(vocab)

	//tokens := Tokenize("Building a website can be done in 10 simple steps:")
	//fmt.Println(tokens)

	tokens := vocab.Tokenize("Mamba is the")
	fmt.Println(tokens)
	fmt.Println(vocab.Detokenize(tokens))
	//fmt.Println(vocab.Detokenize([]int{46, 31834, 310, 253}))

	// some experiments with word vectors
	similarity(m)
	relation(m)

	/*
		tokens := Tokenize(" Suddenly, a magical floppy disk")
		m.SetTemperature(1.2)
		m.SetTokens(tokens)
		m.Run()
	*/
}
