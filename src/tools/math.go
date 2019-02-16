package tools

func DotMulti(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var sum float64 = 0
	for i := 0; i < len(a); i++ {
		sum += a[i]*b[i]
	}
	return sum
}

func NumMulti(a []float64, n float64) float64 {
	var ret float64 = 0
	for i := 0; i < len(a); i++ {
		ret += n * a[i]
	}
	return ret
}