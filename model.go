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

	embedding          *Tensor // token embeddings
	positionalEncoding *Tensor // positional encoding
	lnf                *Tensor // layer normalization
	lnfb               *Tensor // layer normalization bias

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

	m.embedding = GetTensorByName("wte.weight")
	m.positionalEncoding = GetTensorByName("wpe.weight")
	m.lnf = GetTensorByName("ln_f.weight")
	m.lnfb = GetTensorByName("ln_f.bias")

	m.hparams.vocabSize = m.embedding.shape[0]
	m.hparams.dModel = m.embedding.shape[1]
	m.hparams.ctxsize = m.positionalEncoding.shape[0]

	m.emptytoken = 50526 // bos token and eos token (bos= beginning of sentence, eos= end of sentence)
	m.emptytoken = m.vocab.Tokenize("<|endoftext|>")[0]
	fmt.Println("empty token: ", m.emptytoken)

	m.layers = make([]*layer, 0)
	m.hparams.nLayer = 0
	for i := 0; ; i++ {
		if DoesExistTensorByName(fmt.Sprintf("h.%d.attn.c_attn.weight", i)) == false {
			break
		}
		layer := NewLayer(&m.hparams)
		layer.attn_cattn_w = GetTensorByName(fmt.Sprintf("h.%d.attn.c_attn.weight", i))
		layer.attn_cattn_b = GetTensorByName(fmt.Sprintf("h.%d.attn.c_attn.bias", i))
		layer.attn_cproj_w = GetTensorByName(fmt.Sprintf("h.%d.attn.c_proj.weight", i))
		layer.attn_cproj_b = GetTensorByName(fmt.Sprintf("h.%d.attn.c_proj.bias", i))
		layer.mlp_cfc_w = GetTensorByName(fmt.Sprintf("h.%d.mlp.c_fc.weight", i))
		layer.mlp_cfc_b = GetTensorByName(fmt.Sprintf("h.%d.mlp.c_fc.bias", i))
		layer.mlp_cproj_w = GetTensorByName(fmt.Sprintf("h.%d.mlp.c_proj.weight", i))
		layer.mlp_cproj_b = GetTensorByName(fmt.Sprintf("h.%d.mlp.c_proj.bias", i))
		layer.ln1_g = GetTensorByName(fmt.Sprintf("h.%d.ln_1.weight", i))
		layer.ln1_b = GetTensorByName(fmt.Sprintf("h.%d.ln_1.bias", i))
		layer.ln2_g = GetTensorByName(fmt.Sprintf("h.%d.ln_2.weight", i))
		layer.ln2_b = GetTensorByName(fmt.Sprintf("h.%d.ln_2.bias", i))
		//layer. = GetTensorByName(fmt.Sprintf("h.%d.attn.bias", i))
		m.layers = append(m.layers, layer)
		m.hparams.nLayer++
	}

	for _, l := range m.layers {
		l.attn_cattn_w.Transpose()
		l.attn_cproj_w.Transpose()
		l.mlp_cfc_w.Transpose()
		l.mlp_cproj_w.Transpose()
	}

	m.hparams.Init()
	fmt.Printf("%+v\n", m.hparams)

	m.matchlist = make([]match_t, MAXNUMMATCHES)
	m.currwv = make([]float32, m.hparams.dModel)
	return m
}

func (m *Model) SetTemperature(temperature float32) {
	m.temperature = temperature
}

func (m *Model) RunModelForSlot(slot int) {
	// get embedding for given token
	for i := 0; i < m.hparams.dModel; i++ {
		m.currwv[i] = m.embedding.Get2D(i, m.context[slot]) + m.positionalEncoding.Get2D(i, slot)
	}

	for i := 0; i < m.hparams.nLayer; i++ {
		m.layers[i].runLayer(m.currwv[:], slot)
	}

	// Normalize the final result
	Normalize(m.currwv[:], m.currwv[:], m.lnfb.data, m.lnf.data)
	//RMSNorm(x, m.norm_f.data)
}

func (m *Model) Run(input string) {
	m.context = m.vocab.Tokenize(input)
	fmt.Println("context start: ", m.context)
	here := len(m.context)

	// First read the context
	for i := 0; i < here-1; i++ {
		m.RunModelForSlot(i)
		fmt.Printf(m.vocab.DetokenizeSingle(m.context[i]))
	}
	fmt.Printf(m.vocab.sortedVocab[m.context[here-1]])

	// Then generate
	for i := here - 1; i < 1000; i++ {
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

	fmt.Println()
}
