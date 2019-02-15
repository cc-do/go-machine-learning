package tools

// 去重复项，留=
func Cardinality(src []float64) []float64 {
	tgt := make(map[float64]int)
	for i := 0; i < len(src); i++ {
		if _, ok := tgt[src[i]]; !ok {
			tgt[src[i]] = 1
		}
	}
	var ret []float64
	for k := range tgt {
		ret = append(ret, k)
	}
	return ret
}