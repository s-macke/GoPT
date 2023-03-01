package main

import (
	"runtime"
)

func LoadSmall() *Model {
	LoadBin("gpt2_117M.bin")
	runtime.GC()
	m := NewModel(Small)
	runtime.GC()
	return m
}

func LoadLarge() *Model {
	LoadBin("gpt2_1558M.bin")
	runtime.GC()
	m := NewModel(Large)
	runtime.GC()
	return m
}

func main() {
	//debug.SetMemoryLimit(1 << 30) // GB
	LoadVocab("gpt2vocab.txt")
	m := LoadSmall()
	//m := LoadLarge()

	// some experiments with word vectors
	// similarity(m)
	// relation(m)

	tokens := Translate(" Suddenly, a magical floppy disk")
	m.SetTokens(tokens)
	m.Run()
}
