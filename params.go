package main

type GPT2SIZE int64

const (
	Small  GPT2SIZE = 0
	Medium GPT2SIZE = 1
	Large  GPT2SIZE = 2
)

// hyper parameters for each model
const NUMTOKENS = 50257
const MAXNUMMATCHES = 256
const CTXSIZE = 1024
const HEADSIZE = 64
const RSQRT_HEADSIZE = 1. / 8.

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
		// 1558M model
		m.WVSIZE = 1600
		m.NUMLAYERS = 48
		m.NUMHEADS = 25 // WVSIZE / HEADSIZE

	default:
		panic("unknown gpt2size")
	}
}
