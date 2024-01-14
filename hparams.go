package main

type HyperParams struct {
	nLayer    int
	vocabSize int
	dModel    int
	dInner    int
	dtRank    int
}

func (h *HyperParams) Init() {
	expand := 2 // expansion factor
	h.dInner = expand * h.dModel
	h.dtRank = h.dModel / 16
}

// hyper parameters for each model
const MAXNUMMATCHES = 40
