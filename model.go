package main

import (
	"fmt"
)

type match_t struct {
	prob  float32
	token int
}

type Model struct {
	vocab *Vocabulary

	hparams     HyperParams
	temperature float32
	emptytoken  int

	embedding *Tensor // token embeddings
	norm_f    *Tensor

	layers []*layer

	context   []int     // current context as a list of tokens
	currwv    []float32 // current word vector
	matchlist []match_t // TODO
}

//go:noinline
func NewModel(vocabulary *Vocabulary) *Model {
	m := new(Model)
	m.vocab = vocabulary
	m.temperature = 1.0

	m.embedding = GetTensorByName("embedding.weight")
	m.hparams.vocabSize = m.embedding.shape[0]
	m.hparams.dModel = m.embedding.shape[1]

	m.norm_f = GetTensorByName("norm_f.weight")

	m.layers = make([]*layer, 0)
	m.hparams.nLayer = 0
	for i := 0; ; i++ {
		layer := NewLayer()
		if DoesExistTensorByName(fmt.Sprintf("layers.%d.mixer.conv1d.weight", i)) == false {
			break
		}
		m.hparams.nLayer++
		layer.conv1d_weight = GetTensorByName(fmt.Sprintf("layers.%d.mixer.conv1d.weight", i))
		layer.conv1d_bias = GetTensorByName(fmt.Sprintf("layers.%d.mixer.conv1d.bias", i))
		layer.in_proj = GetTensorByName(fmt.Sprintf("layers.%d.mixer.in_proj.weight", i))
		layer.out_proj = GetTensorByName(fmt.Sprintf("layers.%d.mixer.out_proj.weight", i))
		layer.x_proj = GetTensorByName(fmt.Sprintf("layers.%d.mixer.x_proj.weight", i))
		layer.norm = GetTensorByName(fmt.Sprintf("layers.%d.norm.weight", i))
		layer.D = GetTensorByName(fmt.Sprintf("layers.%d.mixer.D", i))
		layer.dt_proj_weight = GetTensorByName(fmt.Sprintf("layers.%d.mixer.dt_proj.weight", i))
		layer.dt_proj_bias = GetTensorByName(fmt.Sprintf("layers.%d.mixer.dt_proj.bias", i))
		layer.A_log = GetTensorByName(fmt.Sprintf("layers.%d.mixer.A_log", i))
		m.layers = append(m.layers, layer)
	}
	m.hparams.Init()
	fmt.Printf("%+v\n", m.hparams)

	m.matchlist = make([]match_t, MAXNUMMATCHES)

	//m.emptytoken = m.vocab.Tokenize("<|endoftext|>")[0]
	m.emptytoken = 0

	m.currwv = make([]float32, m.hparams.dModel)

	return m
}

func (m *Model) SetTemperature(temperature float32) {
	m.temperature = temperature
}

func (m *Model) RunModelForSlot(slot int) {
	x := make([]float32, m.hparams.dModel)

	// get embedding for given token
	for i := 0; i < m.hparams.dModel; i++ {
		m.currwv[i] = m.embedding.Get2D(i, m.context[slot])
	}

	for i := 0; i < m.hparams.nLayer; i++ {
		m.layers[i].runLayer(m.currwv[:], slot)
	}

	/* normalize the final result */
	//Normalize(m.currwv[:], m.currwv[:], m.lnf_b.data, m.lnf_g.data)
	RMSNorm(x, m.norm_f.data)
}

func (m *Model) Run(input string) {
	m.context = m.vocab.Tokenize(input)
	fmt.Println("context start: ", m.context)

	// First read the context
	for i := 0; i < len(m.context); i++ {
		m.RunModelForSlot(i)
		fmt.Printf(m.vocab.DetokenizeSingle(m.context[i]))
	}
	/*
		// Then generate
		fmt.Printf(m.vocab.sortedVocab[m.context[m.here-1]])
		for i := m.here - 1; i < 1000; i++ {
			m.RunModelForSlot(i)

			m.matchToTokens(m.currwv[:], m.matchlist, MAXNUMMATCHES, m.temperature)
			match := pickMatch(m.matchlist, MAXNUMMATCHES, 0.0)
			tok := m.matchlist[match].token
			if tok == m.emptytoken {
				break
			}

			fmt.Printf(m.vocab.DetokenizeSingle(tok))
			m.context = append(m.context, tok)
		}
	*/
	fmt.Println()
}
