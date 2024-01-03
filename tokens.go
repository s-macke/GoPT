package main

import (
	"math"
	"math/rand"
	"sort"
)

// outputs tuples of (dist,token)
// wv is the input vector
// o is the output vector
// num is the number of outputs in the output vector with the highest probability
// temp is the temperature
func (m *Model) matchToTokens(wv []float32, o []match_t, num int, temp float32) {
	t := make([]match_t, m.hparams.NUMTOKENS)

	for i := 0; i < m.hparams.NUMTOKENS; i++ {
		cossim := scalarProduct(wv, m.wte.GetRow2D(i)) // cosine similarity
		t[i].prob = cossim / temp
		t[i].token = i
	}

	sort.SliceStable(t, func(i, j int) bool {
		return t[i].prob > t[j].prob
	})

	// softmax
	max := t[0].prob

	var sum float32 = 0
	for i := 0; i < num; i++ {
		a := float32(math.Exp(float64(t[i].prob - max)))
		t[i].prob = a
		sum += a
	}
	var sumr = 1. / sum
	for i := 0; i < num; i++ {
		o[i].prob = t[i].prob * sumr
		o[i].token = t[i].token
	}
}

// pickmatch picks a random match from the match list.
// tokenflags[] allows for some token-specific options.
// list is the list of matches
// sz is the number of matches in the list
// minp is the minimum probability to consider.
// returns the index of the match chosen.
func pickmatch_(list []match_t, sz int, minp float32) int {
	var i int

	if list[0].prob < minp || list[0].prob > 0.98 {
		return 0
	}
	a := rand.Float32()
	for i = 0; i < sz; i++ {
		p := list[i].prob
		if p < minp {
			i = 0
			p = list[i].prob
		}
		a -= p
		if a <= 0 {
			return i
		}
	}
	return 0
}

func pickmatch(list []match_t, sz int, minp float32) int {
	i := pickmatch_(list, sz, minp)
	return i
}
