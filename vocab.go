package main

import (
	"strings"
)

var vocab []string

// translates a string into a list of tokens
func Translate(str string) (result []int) {
	for {
		largestmatchidx := -1
		largestsize := -1
		for i := 0; i < len(vocab); i++ {
			if strings.HasPrefix(str, vocab[i]) {
				if len(vocab[i]) > largestsize {
					largestsize = len(vocab[i])
					largestmatchidx = i
				}
			}
		}
		if largestmatchidx == -1 {
			panic("No match")
		}

		result = append(result, largestmatchidx)
		str = str[len(vocab[largestmatchidx]):]
		if len(str) == 0 {
			//fmt.Println(str, result)
			return
		}
	}

}
