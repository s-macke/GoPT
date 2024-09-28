package main

import (
	"math"
)

type layer struct {
	hparams *HyperParams

	// read only
	ln1_b, ln1_g, ln2_b, ln2_g *Tensor
	mlp_cfc_b, mlp_cfc_w       *Tensor
	mlp_cproj_b, mlp_cproj_w   *Tensor
	attn_cattn_b, attn_cattn_w *Tensor
	attn_cproj_b, attn_cproj_w *Tensor

	// read/write
	key   []float32 // key vectors
	value []float32 // value vectors
}

func NewLayer(hparams *HyperParams) *layer {
	return &layer{
		hparams: hparams,
		key:     make([]float32, hparams.dModel*hparams.ctxsize),
		value:   make([]float32, hparams.dModel*hparams.ctxsize),
	}
}

// runLayer runs a single layer of the transformer
// x is the input vector, the word embedding of the current token
// slot is the index of the current token in the context
func (l *layer) runLayer(x []float32, slot int) {
	const RSQRT_HEADSIZE = 1. / 8.
	var query = make([]float32, l.hparams.dModel) // query vectors are only needed locally
	var tmp = make([]float32, l.hparams.dModel)   // tmp space for operations
	var xn = make([]float32, l.hparams.dModel)

	Normalize(xn[:], x, l.ln1_b.data, l.ln1_g.data)

	// produce query/key/value vectors for this slot
	b := l.attn_cattn_b
	w := l.attn_cattn_w
	for i := 0; i < l.hparams.dModel*3; i++ {
		a := b.data[i] + ScalarProduct(xn[:], w.GetRow2D(i))
		if i < l.hparams.dModel {
			query[i] = a
		} else if i < l.hparams.dModel*2 {
			l.key[slot*l.hparams.dModel+(i-l.hparams.dModel)] = a
		} else {
			l.value[(i-l.hparams.dModel*2)*l.hparams.ctxsize+slot] = a
		}
	}

	// run for each attention head
	att := make([]float32, slot+1)
	for h := 0; h < l.hparams.NUMHEADS; h++ {
		// query * keys = attentions
		for i := 0; i <= slot; i++ {
			a := ScalarProduct(
				query[h*l.hparams.HEADSIZE:(h+1)*l.hparams.HEADSIZE],
				l.key[i*l.hparams.dModel+h*l.hparams.HEADSIZE:i*l.hparams.dModel+(h+1)*l.hparams.HEADSIZE])
			att[i] = a * RSQRT_HEADSIZE
		}

		// softmax attentions to make them sum up to 1.0
		var softmax = att[0]
		for i := 1; i <= slot; i++ {
			if att[i] > softmax {
				softmax = att[i]
			}
		}

		var sum float32 = 0.
		for i := 0; i <= slot; i++ {
			a := float32(math.Exp(float64(att[i]) - float64(softmax)))
			att[i] = a
			sum += a
		}
		var sumr = 1. / sum
		for i := 0; i <= slot; i++ {
			att[i] *= sumr
		}

		// apply attentions to values
		for j := 0; j < l.hparams.HEADSIZE; j++ {
			offset := (j + h*l.hparams.HEADSIZE) * l.hparams.ctxsize
			tmp[h*l.hparams.HEADSIZE+j] = ScalarProduct(att[0:slot+1], l.value[offset:offset+(slot+1)])
		}
	}

	// projection (WVSIZExWVSIZE)
	w = l.attn_cproj_w
	b = l.attn_cproj_b
	for i := 0; i < l.hparams.dModel; i++ {
		x[i] += b.data[i] + ScalarProduct(tmp[:], w.data[l.hparams.dModel*i:l.hparams.dModel*(i+1)])
	}

	// normalize again
	Normalize(xn[:], x, l.ln2_b.data, l.ln2_g.data)

	// multilayer perceptron (dModel -> dModel*4 -> dModel)
	w = l.mlp_cfc_w
	b = l.mlp_cfc_b
	mlp := make([]float32, l.hparams.dModel*4)

	for i := 0; i < l.hparams.dModel*4; i++ {
		a := b.data[i] + ScalarProduct(xn[:], w.data[l.hparams.dModel*i:l.hparams.dModel*(i+1)])
		a = 0.5 * a * (1. + float32(math.Tanh(float64(0.7978845676080871*(a+0.044715*a*a*a))))) // gelu2 ?
		mlp[i] = a
	}

	w = l.mlp_cproj_w
	b = l.mlp_cproj_b
	for i := 0; i < l.hparams.dModel; i++ {
		x[i] += b.data[i] + ScalarProduct(mlp[:], w.data[l.hparams.dModel*4*i:l.hparams.dModel*4*(i+1)])
	}
}
