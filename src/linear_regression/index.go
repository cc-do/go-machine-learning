package linear_regression

import (
	"tools"
	"fmt"
	"math"
)

type LinReg struct {
	X [][]float64		// 训练样本
	Y []float64			// 预测结果
	Q []float64			// 标签
	M int				// 训练样本数
	N int				// 特征数（样本数+1）
}

func (lr *LinReg) SetDataset(samples [][]float64, results []float64) {
	lr.X = samples
	lr.Y = results
	lr.M = len(lr.Y)
	lr.N = len(lr.X[0])
	for t := 0; t < lr.N + 1; t++ {
		lr.Q = append(lr.Q, 1)
	}

	// 标准化样本数据
	lr.X = tools.Normalize(lr.X)
	fmt.Println("经过标准化之后的样本：")
	for i := 0; i < lr.M; i++ {
		fmt.Println(lr.X[i])
	}
}

func (lr *LinReg) H(i int) float64 {
	sum := lr.Q[0]
	for j := 1; j < lr.N + 1; j++ {
		sum += lr.Q[j] * lr.X[i][j - 1]
	}
	return sum
}

func (lr *LinReg) J(j int) float64 {
	var sum float64 = 0
	for i := 0; i < lr.M; i++ {
		// X0 = 1 或者使用指定的样例
		var xij float64 = 1
		if j != 0 {
			xij = lr.X[i][j - 1]
		}
		sum += (lr.H(i) -lr. Y[i]) * xij
	}
	return sum / float64(lr.M)
}

func (lr *LinReg) Learn(maxLoop int, accuracy float64, velocity float64) {
	var acc float64 = 1
	for t := 0; t < maxLoop && acc > accuracy; t++ {
		acc = 1
		for j := 0; j < lr.N + 1; j++ {
			cost := lr.J(j)
			// 找出绝对值最小的误差作为精度
			if acc > math.Abs(cost) {
				acc = math.Abs(cost)
			}
			lr.Q[j] = lr.Q[j] - velocity * cost
		}
		fmt.Printf("精度值：%f\n", acc)
	}
	fmt.Println("最终特征数组: ", lr.Q)
}

func (lr *LinReg) Test() {
	for i := 0; i < lr.M; i++ {
		fmt.Printf("预测结果：%f，模型计算结果：%f\n", lr.Y[i], lr.H(i))
	}
}