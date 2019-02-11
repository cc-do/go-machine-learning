package linear_regression

import (
	"common"
	"math"
)

func H(i int) float64 {
	g := common.GetGlbVar()
	var sum float64
	for j := 0; j < g.N; j++ {
		sum += g.Q[j] * g.X[i][j]
	}
	return sum
}

func J(j int) float64 {
	g := common.GetGlbVar()
	var sum float64 = 0
	for i := 0; i < g.M; i++ {
		sum += math.Abs(H(i) - g.Y[i]) * g.X[i][j]
	}
	return sum / float64(g.M)
}