package main

import (
	"math"
)

type layer struct {
	// read only
	ln1_b, ln1_g, ln2_b, ln2_g *tensor
	mlp_cfc_b, mlp_cfc_w       *tensor
	mlp_cproj_b, mlp_cproj_w   *tensor
	attn_cattn_b, attn_cattn_w *tensor
	attn_cproj_b, attn_cproj_w *tensor

	WVSIZE   int
	CTXSIZE  int
	NUMHEADS int
	// read/write
	key   []float32 // key vectors
	value []float32 // value vectors
}

func newLayer(WVSIZE int, CTXSIZE int, NUMHEADS int) *layer {
	return &layer{
		WVSIZE:   WVSIZE,
		CTXSIZE:  CTXSIZE,
		NUMHEADS: NUMHEADS,
		key:      make([]float32, WVSIZE*CTXSIZE),
		value:    make([]float32, WVSIZE*CTXSIZE),
	}
}

// runLayer runs a single layer of the transformer
// x is the input vector, the word embedding of the current token
// slot is the index of the current token in the context
func (l *layer) runLayer(x []float32, slot int) {
	var query = make([]float32, l.WVSIZE) // query vectors are only needed locally
	var tmp = make([]float32, l.WVSIZE)   // tmp space for operations
	var xn = make([]float32, l.WVSIZE)

	Normalize(xn[:], x, l.ln1_b.data, l.ln1_g.data)

	// produce query/key/value vectors for this slot

	b := l.attn_cattn_b
	w := l.attn_cattn_w
	for i := 0; i < l.WVSIZE*3; i++ {
		a := b.data[i] + scalarProduct(xn[:], w.GetRow2D(i))

		if i < l.WVSIZE {
			query[i] = a
		} else if i < l.WVSIZE*2 {
			l.key[slot*l.WVSIZE+(i-l.WVSIZE)] = a
		} else {
			l.value[(i-l.WVSIZE*2)*CTXSIZE+slot] = a
		}
	}

	// run for each attention head
	att := make([]float32, slot+1)
	for h := 0; h < l.NUMHEADS; h++ {
		// query * keys = attentions
		for i := 0; i <= slot; i++ {
			a := scalarProduct(query[h*HEADSIZE:(h+1)*HEADSIZE], l.key[i*l.WVSIZE+h*HEADSIZE:i*l.WVSIZE+(h+1)*HEADSIZE])
			att[i] = a * RSQRT_HEADSIZE
		}

		// softmax attentions to make them sum up to 1.0
		var max = att[0]
		for i := 1; i <= slot; i++ {
			if att[i] > max {
				max = att[i]
			}
		}

		var sum float32 = 0.
		for i := 0; i <= slot; i++ {
			a := float32(math.Exp(float64(att[i]) - float64(max)))
			att[i] = a
			sum += a
		}
		var sumr = 1. / sum
		for i := 0; i <= slot; i++ {
			att[i] *= sumr
		}

		// apply attentions to values
		for j := 0; j < HEADSIZE; j++ {
			offset := (j + h*HEADSIZE) * CTXSIZE
			tmp[h*HEADSIZE+j] = scalarProduct(att[0:slot+1], l.value[offset:offset+(slot+1)])
		}
	}

	// projection (WVSIZExWVSIZE)
	w = l.attn_cproj_w
	b = l.attn_cproj_b
	for i := 0; i < l.WVSIZE; i++ {
		x[i] += b.data[i] + scalarProduct(tmp[:], w.data[l.WVSIZE*i:l.WVSIZE*(i+1)])
	}

	// normalize again
	Normalize(xn[:], x, l.ln2_b.data, l.ln2_g.data)

	// multilayer perceptron (WVSIZE -> WVSIZE*4 -> WVSIZE)
	w = l.mlp_cfc_w
	b = l.mlp_cfc_b
	mlp := make([]float32, l.WVSIZE*4)

	for i := 0; i < l.WVSIZE*4; i++ {
		a := b.data[i] + scalarProduct(xn[:], w.data[l.WVSIZE*i:l.WVSIZE*(i+1)])
		a = 0.5 * a * (1. + float32(math.Tanh(float64(0.7978845676080871*(a+0.044715*a*a*a))))) // gelu2 ?
		mlp[i] = a
	}

	w = l.mlp_cproj_w
	b = l.mlp_cproj_b
	for i := 0; i < l.WVSIZE; i++ {
		x[i] += b.data[i] + scalarProduct(mlp[:], w.data[l.WVSIZE*4*i:l.WVSIZE*4*(i+1)])
	}
}
