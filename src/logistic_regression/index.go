package logistic_regression

import (
	"linear_regression"
	"math"
	"fmt"
)

type LogReg struct {
	linear_regression.LinReg
}

func (lr *LogReg) G(i int) float64 {
	return 1/ (1 + math.Exp(-lr.H(i)))
}

func (lr *LogReg) J(j int) float64 {
	var sum float64 = 0
	for i := 0; i < lr.M; i++ {
		// X0 = 1 或者使用指定的样例
		var xij float64 = 1
		if j != 0 {
			xij = lr.X[i][j - 1]
		}
		sum += (lr.Y[i] - lr.G(i)) * xij
	}
	return sum / float64(lr.M)
}

func (lr *LogReg) Learn(maxLoop int, accuracy float64, velocity float64) {
	var acc float64 = 1
	for t := 0; t < maxLoop && acc > accuracy; t++ {
		acc = 1
		for j := 0; j < lr.N + 1; j++ {
			cost := lr.J(j)
			// 找出绝对值最小的误差作为精度
			if acc > math.Abs(cost) {
				acc = math.Abs(cost)
			}
			lr.Q[j] = lr.Q[j] + velocity * cost
		}
		fmt.Printf("精度值：%f\n", acc)
	}
	fmt.Println("最终特征数组: ", lr.Q)
}

func (lr *LogReg) Test() {
	for i := 0; i < lr.M; i++ {
		fmt.Printf("预测结果：%f，模型计算结果：%f\n", lr.Y[i], lr.G(i))
	}
}