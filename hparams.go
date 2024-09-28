package main

type HyperParams struct {
	nLayer    int
	vocabSize int
	dModel    int
	ctxsize   int
	NUMHEADS  int
	HEADSIZE  int
}

func (h *HyperParams) Init() {
	// WVSIZE = dModel = 768
	h.NUMHEADS = 12 // number of attention heads WVSIZE / HEADSIZE
	h.HEADSIZE = 64
}

// hyper parameters for each model
const MAXNUMMATCHES = 40
