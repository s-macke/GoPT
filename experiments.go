package main

import (
	"fmt"
	"math"
)

func tokenizeSingle(m *Model, str string) int {
	tokens := m.vocab.Tokenize(str)
	if len(tokens) > 1 {
		panic("More than one token for string " + str)
	}
	return tokens[0]
}

func (m *Model) Add(token1, token2, token3 int) {
	//token1 - token2 + token3
	wv := make([]float32, m.hparams.WVSIZE)
	for i := 0; i < m.hparams.WVSIZE; i++ {
		wv[i] = m.embedding.Get2D(i, token1) - m.embedding.Get2D(i, token2) + m.embedding.Get2D(i, token3)
	}
	m.matchToTokens(wv, m.matchlist, 20, 1.)
	for i := 0; i < 10; i++ {
		fmt.Println(m.vocab.DetokenizeSingle(m.matchlist[i].token))
	}
}

// https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85
func (m *Model) wordMath(str1, str2, str3 string) {
	fmt.Println("-----", str1, str2, str3)
	m.Add(tokenizeSingle(m, str1), tokenizeSingle(m, str2), tokenizeSingle(m, str3))
	fmt.Println("-----")
}

func relation(m *Model) {
	m.wordMath(" king", " man", " woman")
	m.wordMath(" bigger", " big", " cold")
	m.wordMath(" ran", " walked", " walk")
}

// Compare computes the cosine similarity between two vectors
func compare(token1, token2 int, m *Model) float32 {
	var dist1 float32 = 0
	var dist2 float32 = 0
	var sp float32 = 0
	for i := 0; i < m.hparams.WVSIZE; i++ {
		//diff := m.wte.Get2D(i, token1) - m.wte.Get2D(i, token2)
		//dist1 += diff * diff
		//sp += m.wte.Get2D(token1, i) * m.wte.Get2D(token2, i)
		sp += m.embedding.Get2D(i, token1) * m.embedding.Get2D(i, token2)
		dist1 += m.embedding.Get2D(i, token1) * m.embedding.Get2D(i, token1)
		dist2 += m.embedding.Get2D(i, token2) * m.embedding.Get2D(i, token2)
	}
	//return sp
	return sp / (float32(math.Sqrt(float64(dist1))) * float32(math.Sqrt(float64(dist2))))
	//return float32(math.Sqrt(float64(dist1)))
}

func similarity(m *Model) {
	tokens := make([]int, 0)
	tokens = append(tokens, tokenizeSingle(m, " Blue"))
	tokens = append(tokens, tokenizeSingle(m, " Green"))
	tokens = append(tokens, tokenizeSingle(m, " Red"))
	tokens = append(tokens, tokenizeSingle(m, "1"))
	tokens = append(tokens, tokenizeSingle(m, "2"))
	tokens = append(tokens, tokenizeSingle(m, "3"))
	for i := 0; i < len(tokens); i++ {
		for j := 0; j < len(tokens); j++ {
			fmt.Printf("%7.3f ", compare(tokens[i], tokens[j], m))
		}
		fmt.Println()
	}

}
