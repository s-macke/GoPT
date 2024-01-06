package main

import (
	"encoding/json"
	"io"
	"os"
	"strings"
)

type Vocabulary struct {
	vocab       map[string]int
	sortedVocab []string
}

func NewVocabulary(filename string) *Vocabulary {
	vocabulary := new(Vocabulary)
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	bytes, err := io.ReadAll(f)
	if err != nil {
		panic(err)
	}
	var v map[string]int

	err = json.Unmarshal(bytes, &v)
	if err != nil {
		panic(err)
	}

	vocabulary.vocab = make(map[string]int)
	vocabulary.sortedVocab = make([]string, len(v))
	for token, index := range v {
		tok := strings.Replace(token, "Ġ", " ", -1)
		vocabulary.vocab[tok] = index
		vocabulary.sortedVocab[index] = tok
	}
	return vocabulary
}

// Tokenize translates a string into a list of tokens
func (v *Vocabulary) Tokenize(str string) (tokens []int) {

forEachToken:
	for {
		if len(str) == 0 {
			break
		}
		for i := 0; i < len(v.vocab); i++ {
			if strings.HasPrefix(str, v.sortedVocab[i]) {
				tokens = append(tokens, i)
				str = str[len(v.sortedVocab[i]):]
				continue forEachToken
			}
		}
		panic("No match")
	}

	// first tokenization done, now merge tokens
	for {
		bestTokenId := -1
		//bestScore := -1
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			testToken := v.sortedVocab[tokens[i]] + v.sortedVocab[tokens[i+1]]
			if tokenId, ok := v.vocab[testToken]; ok {
				//if len(testToken) > bestScore {
				bestTokenId = tokenId
				//bestScore = len(testToken)
				bestIdx = i
				//}
			}
		}
		if bestTokenId == -1 {
			break
		}
		// merge two tokens
		tokens = append(tokens[:bestIdx], append([]int{bestTokenId}, tokens[bestIdx+2:]...)...)
	}
	return
}

func (v *Vocabulary) DetokenizeSingle(token int) string {
	return v.sortedVocab[token]
}

func (v *Vocabulary) Detokenize(tokens []int) string {
	var sb strings.Builder
	for _, token := range tokens {
		sb.WriteString(v.sortedVocab[token])
	}
	return sb.String()
}
