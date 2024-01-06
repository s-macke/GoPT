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

	hparams HyperParams
	context []int
	here    int

	emptytoken int
	embedding  *Tensor // token embeddings
	norm_f     *Tensor

	layers []*layer

	currwv []float32

	temperature float32

	matchlist []match_t
}

//go:noinline
func NewModel(vocabulary *Vocabulary) *Model {
	m := new(Model)
	m.vocab = vocabulary

	m.embedding = FindTensor("embedding.weight")
	m.hparams.NUMTOKENS = m.embedding.shape[0]
	m.hparams.WVSIZE = m.embedding.shape[1]
	m.embedding.Transpose()

	m.norm_f = FindTensor("norm_f.weight")

	m.layers = make([]*layer, 0)
	m.hparams.NUMLAYERS = 0
	for i := 0; ; i++ {
		layer := newLayer()
		t := FindTensor(fmt.Sprintf("layers.%d.mixer.conv1d.bias", i))
		if t == nil {
			break
		}
		m.hparams.NUMLAYERS++
		layer.conv1d_bias = FindTensor(fmt.Sprintf("layers.%d.mixer.conv1d.bias", i))
		m.layers = append(m.layers, layer)
	}
	fmt.Printf("%+v\n", m.hparams)

	/*
		for i := 0; i < m.hparams.NUMLAYERS; i++ {
			m.layers[i] = newLayer(m.hparams.WVSIZE, m.hparams.CTXSIZE, m.hparams.NUMHEADS)
		}
		m.currwv = make([]float32, m.hparams.WVSIZE)
	*/
	m.temperature = 1.0
	m.matchlist = make([]match_t, MAXNUMMATCHES)

	//m.emptytoken = Tokenize("<|endoftext|>")[0]
	//m.emptytoken = 0 // TODO

	//m.SetTokens([]int{})

	/*
		for layeri := 0; layeri < m.hparams.NUMLAYERS; layeri++ {
			m.layers[layeri].ln1_g = FindTensor(fmt.Sprintf("h%d/ln_1/g", layeri))
			m.layers[layeri].ln1_b = FindTensor(fmt.Sprintf("h%d/ln_1/b", layeri))
			m.layers[layeri].ln2_g = FindTensor(fmt.Sprintf("h%d/ln_2/g", layeri))
			m.layers[layeri].ln2_b = FindTensor(fmt.Sprintf("h%d/ln_2/b", layeri))
			m.layers[layeri].mlp_cfc_w = FindTensor(fmt.Sprintf("h%d/mlp/c_fc/w", layeri))
			m.layers[layeri].mlp_cfc_b = FindTensor(fmt.Sprintf("h%d/mlp/c_fc/b", layeri))
			m.layers[layeri].mlp_cproj_w = FindTensor(fmt.Sprintf("h%d/mlp/c_proj/w", layeri))
			m.layers[layeri].mlp_cproj_b = FindTensor(fmt.Sprintf("h%d/mlp/c_proj/b", layeri))
			m.layers[layeri].attn_cproj_w = FindTensor(fmt.Sprintf("h%d/attn/c_proj/w", layeri))
			m.layers[layeri].attn_cproj_b = FindTensor(fmt.Sprintf("h%d/attn/c_proj/b", layeri))
			m.layers[layeri].attn_cattn_w = FindTensor(fmt.Sprintf("h%d/attn/c_attn/w", layeri))
			m.layers[layeri].attn_cattn_b = FindTensor(fmt.Sprintf("h%d/attn/c_attn/b", layeri))
		}

		for _, l := range m.layers {
			l.attn_cattn_w.Transpose()
			l.attn_cproj_w.Transpose()
			l.mlp_cfc_w.Transpose()
			l.mlp_cproj_w.Transpose()
		}
	*/
	return m
}

func (m *Model) SetTemperature(temperature float32) {
	m.temperature = temperature
}

func (m *Model) SetTokens(tokens []int) {
	/*
		for i := 0; i < m.hparams.CTXSIZE; i++ {
			m.context[i] = m.emptytoken
		}
		for i, t := range tokens {
			m.context[i] = t
		}
		m.here = len(tokens)
	*/
}

func (m *Model) RunModelForSlot(slot int) {
	/*
		// token vector with positional encoding
		for i := 0; i < m.hparams.WVSIZE; i++ {
			m.currwv[i] = m.wte.Get2D(i, m.context[slot]) + m.wpe.Get2D(i, slot)
		}
		for i := 0; i < m.hparams.NUMLAYERS; i++ {
			m.layers[i].runLayer(m.currwv[:], slot)
		}

	*/
	/* normalize the final result */
	//Normalize(m.currwv[:], m.currwv[:], m.lnf_b.data, m.lnf_g.data)
}

func (m *Model) Run() {
	// First read the context
	for i := 0; i < m.here-1; i++ {
		m.RunModelForSlot(i)
		fmt.Printf(m.vocab.DetokenizeSingle(m.context[i]))
	}
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
		m.context[i+1] = tok
	}
}
