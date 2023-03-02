package main

import (
	"os"
	"runtime"
	"runtime/debug"
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
	debug.SetMemoryLimit(1 << 30) // GB
	LoadVocab("gpt2vocab.txt")
	var m *Model

	if len(os.Args) > 1 && os.Args[1] == "large" {
		m = LoadLarge()
	} else {
		m = LoadSmall()
	}

	// some experiments with word vectors
	// similarity(m)
	// relation(m)

	tokens := Translate(" Suddenly, a magical floppy disk")
	m.SetTemperature(1.2)
	m.SetTokens(tokens)
	m.Run()
}
