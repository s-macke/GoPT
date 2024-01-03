package main

type HyperParams struct {
	NUMTOKENS int
	CTXSIZE   int
	WVSIZE    int
	NUMLAYERS int
	MULT      int
	NUMHEADS  int
	N_ROT     int
	f16       int
}

// hyper parameters for each model
const MAXNUMMATCHES = 40
