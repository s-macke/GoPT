package main

import (
	"fmt"
	"runtime"
)

func main() {
	//LoadSafetensors("llama3.2.safetensors")
	//os.Exit(1)
	LoadSafetensors("model.safetensors")
	vocab := NewVocabulary("vocab.json")
	fmt.Println("Vocabulary size:", len(vocab.vocab))

	runtime.GC()
	for _, t := range bmodel.tensors {
		t.ToFloat32()
	}
	runtime.GC()

	m := NewModel(vocab)

	//tokens := Tokenize("Building a website can be done in 10 simple steps:")
	//fmt.Println(tokens)

	//tokens := m.vocab.Tokenize("<|endoftext|>")
	//fmt.Println(tokens) // should be 50256

	// some experiments with word vectors
	//similarity(m)
	//relation(m)

	//tokens := vocab.Tokenize(" Suddenly, a magical floppy disk")
	//fmt.Println(tokens)

	m.SetTemperature(1.0)
	m.Run(" Suddenly, a magical floppy disk")
}
