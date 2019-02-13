package newton_method

import "common"

func J(j int) float64 {
	g := common.GetGlbVar()
	var sum float64 = 0
	// Hessian矩阵的逆矩阵乘以f(x)的梯度
	return sum / float64(g.M)
}
