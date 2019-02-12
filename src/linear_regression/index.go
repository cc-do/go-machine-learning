package linear_regression

import (
	"common"
)

func H(i int) float64 {
	g := common.GetGlbVar()
	sum := g.Q[0]
	for j := 1; j < g.N + 1; j++ {
		sum += g.Q[j] * g.X[i][j - 1]
	}
	return sum
}

func J(j int) float64 {
	g := common.GetGlbVar()
	var sum float64 = 0
	for i := 0; i < g.M; i++ {
		// X0 = 1 或者使用指定的样例
		var xij float64 = 1
		if j != 0 {
			xij = g.X[i][j - 1]
		}
		sum += (H(i) - g.Y[i]) * xij
	}
	return sum / float64(g.M)
}