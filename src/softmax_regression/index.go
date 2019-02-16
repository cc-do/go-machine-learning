package softmax_regression

import (
	"math"
	"linear_regression"
	"tools"
	"fmt"
)

type SmxReg struct {
	linear_regression.LinReg
	Q [][]float64			// 标签
	K []float64				// 样例基数
	R float64				// 权重衰减项
}

func (sr *SmxReg) SetDataset(samples [][]float64, results []float64) {
	sr.X = samples
	sr.Y = results
	sr.M = len(sr.Y)
	sr.N = len(sr.X[0])
	sr.R = 0.02
	// 计算输出空间的离散样本数
	for t := 0; t < sr.M; t++ {
		sr.K = tools.Cardinality(sr.Y)
	}
	// 生成NxK的参数矩阵
	for i := 0; i < len(sr.K); i++ {
		var tmpAry []float64
		for j := 0; j < sr.N + 1; j++ {
			tmpAry = append(tmpAry, 1)
		}
		sr.Q = append(sr.Q, tmpAry)
	}

	// 标准化样本数据
	sr.X = tools.Normalize(sr.X)
	fmt.Println("经过标准化之后的样本：")
	for i := 0; i < sr.M; i++ {
		fmt.Println(sr.X[i])
	}
}

func (sr *SmxReg) H(i int, j int) float64 {
	tmpX := append([]float64{1}, sr.X[i]...)
	return math.Exp(tools.DotMulti(tmpX, sr.Q[j]))
}

func (sr *SmxReg) G(i int, j int) float64 {
	tmpX := append([]float64{1}, sr.X[i]...)
	numerator := math.Exp(tools.DotMulti(tmpX, sr.Q[j]))
	var denominator float64 = 0
	for t := 0; t < len(sr.K); t++ {
		denominator += math.Exp(tools.DotMulti(tmpX, sr.Q[t]))
	}
	return numerator / denominator
}

func (sr *SmxReg) J(j int) float64 {
	var sum float64 = 0
	for i := 0; i < sr.M; i++ {
		// X0 = 1 或者使用指定的样例
		var xij float64 = 1
		if j != 0 {
			xij = sr.X[i][j - 1]
		}

		var sgl float64 = 0
		if sr.Y[i] == sr.K[j] {
			sgl = 1
		}
		sum += (sgl - sr.G(i, j)) * xij
	}
	return -sum / float64(sr.M) + tools.NumMulti(sr.Q[j], sr.R)
}

func (sr *SmxReg) Learn(maxLoop int, accuracy float64, velocity float64) {
	acc := math.MaxFloat64
	for t := 0; t < maxLoop && acc > accuracy; t++ {
		acc = math.MaxFloat64
		for j := 0; j < len(sr.Q); j++ {
			for i := 0; i < sr.N + 1; i++ {
				cost := sr.J(j)
				// 找出绝对值最小的误差作为精度
				if acc > math.Abs(cost) {
					acc = math.Abs(cost)
				}
				sr.Q[j][i] = sr.Q[j][i] - velocity * cost
			}
		}
		fmt.Printf("精度值：%.10f\n", acc)
	}
	fmt.Println("最终特征数组: [")
	for i := 0; i < len(sr.Q); i++ {
		fmt.Printf("\t")
		fmt.Println(sr.Q[i])
	}
	fmt.Println("]")
}

func (sr *SmxReg) Test() {
	for i := 0; i < sr.M; i++ {
		fmt.Printf("预测结果：%f，", sr.Y[i])
		jCur := 0
		for j := 0; j < len(sr.K); j++ {
			if sr.K[j] == sr.Y[i] {
				jCur = j
				fmt.Printf("计算结果：%f，", sr.H(i, jCur))
			}
		}
		fmt.Print("其他结果：")
		for j := 0; j < len(sr.K); j++ {
			if j == jCur {
				continue
			}
			fmt.Printf("%f:%f，", sr.K[j], sr.H(i, j))
		}
		fmt.Println()
	}
}