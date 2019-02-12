package logistic_regression

import (
	"common"
	"linear_regression"
	"math"
)

func G(i int) float64 {
	return 1/ (1 + math.Exp(-linear_regression.H(i)))
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
		sum += (G(i) - g.Y[i]) * xij
	}
	return sum / float64(g.M)
}