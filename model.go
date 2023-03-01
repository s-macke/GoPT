package main

import (
	"fmt"
)

/*
// not used yet! we'll support several simultaneous contexts in the future
typedef struct {
  token_t*in;
  float*k;
  float*v;
  int lgt;
  int validlgt;
} context_t;
*/

type GPT2SIZE int64

const (
	Small  GPT2SIZE = 0
	Medium GPT2SIZE = 1
	Large  GPT2SIZE = 2
)

type match_t struct {
	prob  float32
	token int
}

// hparams for each model
const NUMTOKENS = 50257
const MAXNUMMATCHES = 256
const CTXSIZE = 1024
const HEADSIZE = 64
const RSQRT_HEADSIZE = 1. / 8.

type Model struct {
	WVSIZE    int
	NUMLAYERS int
	NUMHEADS  int

	context [CTXSIZE]int
	here    int

	emptytoken int
	lnf_g      *tensor
	lnf_b      *tensor
	wte        *tensor // token's wordvector (wte)
	wpe        *tensor // positional salt (wpe)
	layers     []*layer

	currwv []float32

	temperature float32

	matchlist []match_t
}

//go:noinline
func NewModel(gpt2size GPT2SIZE) *Model {
	m := new(Model)
	m.SetParameters(gpt2size)
	m.layers = make([]*layer, m.NUMLAYERS)
	for i := 0; i < m.NUMLAYERS; i++ {
		m.layers[i] = newLayer(m.WVSIZE, CTXSIZE, m.NUMHEADS)
	}
	m.currwv = make([]float32, m.WVSIZE)

	m.temperature = 1.2

	m.matchlist = make([]match_t, MAXNUMMATCHES)

	m.emptytoken = Translate("<|endoftext|>")[0]
	fmt.Println("emptytoken", m.emptytoken)

	m.SetTokens([]int{})

	m.lnf_b = FindTensor("ln_f/b")
	m.lnf_g = FindTensor("ln_f/g")
	m.wte = FindTensor("wte")
	m.wpe = FindTensor("wpe")

	for layeri := 0; layeri < m.NUMLAYERS; layeri++ {
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

	return m
}

func (m *Model) SetParameters(gpt2size GPT2SIZE) {

	switch gpt2size {
	case Small:
		// 117M model
		m.WVSIZE = 768
		m.NUMLAYERS = 12
		m.NUMHEADS = 12 // WVSIZE / HEADSIZE

	case Medium:
		m.WVSIZE = 1024
		m.NUMLAYERS = 24
		m.NUMHEADS = 16 // WVSIZE / HEADSIZE

	case Large:
		m.WVSIZE = 1600
		m.NUMLAYERS = 48
		m.NUMHEADS = 25 // WVSIZE / HEADSIZE

	default:
		panic("unknown gpt2size")

	}
}

func (m *Model) SetTokens(tokens []int) {
	for i := 0; i < CTXSIZE; i++ {
		m.context[i] = m.emptytoken
	}
	for i, t := range tokens {
		m.context[i] = t
	}
	m.here = len(tokens)
}

func (m *Model) Add(token1, token2, token3 int) {
	//token1-token2+token3
	wv := make([]float32, m.WVSIZE)
	for i := 0; i < m.WVSIZE; i++ {
		wv[i] = m.wte.Get2D(i, token1) - m.wte.Get2D(i, token2) + m.wte.Get2D(i, token3)
	}
	m.matchToTokens(wv, m.matchlist, 20, 1.)
	for i := 0; i < 20; i++ {
		fmt.Println(vocab[m.matchlist[i].token])
	}
}

func (m *Model) RunModelForSlot(slot int) {
	// token vector with positional encoding
	for i := 0; i < m.WVSIZE; i++ {
		m.currwv[i] = m.wte.Get2D(i, m.context[slot]) + m.wpe.Get2D(i, slot)
	}
	for i := 0; i < m.NUMLAYERS; i++ {
		m.layers[i].runLayer(m.currwv[:], slot)
	}
	/* normalize the final result */
	Normalize(m.currwv[:], m.currwv[:], m.lnf_b.data, m.lnf_g.data)
}

func (m *Model) Run() {
	for i := 0; i < m.here-1; i++ {
		m.RunModelForSlot(i)
		fmt.Printf(vocab[m.context[i]])
	}
	fmt.Printf(vocab[m.context[m.here-1]])
	for i := m.here - 1; i < 1000; i++ {
		m.RunModelForSlot(i)
		m.matchToTokens(m.currwv[:], m.matchlist, 40, 1.2)
		match := pickmatch(m.matchlist, 40, 0.0)
		tok := m.matchlist[match].token
		if tok == m.emptytoken {
			break
		}

		fmt.Printf(vocab[tok])
		m.context[i+1] = tok
	}
}
