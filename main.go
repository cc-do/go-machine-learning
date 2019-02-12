package main

import (
	"common"
	"fmt"
	"linear_regression"
	"log"
	"os"
	"tools"
)

func main() {
	// 读取训练数据集
	var err error
	var samples [][]float64
	var results []float64
	if samples, results, err = tools.LoadDataset("数据集/heart.csv", "chol"); err != nil {
		log.Fatal(err)
		os.Exit(-1)
	}
	if len(samples) == 0 || len(results) == 0 {
		os.Exit(-1)
	}
	if len(samples) != len(results) {
		log.Fatal("样本数量和结果数量不同")
		os.Exit(2)
	}
	g := common.SetGlbVar(samples, results)

	// 标准化样本数据
	for j := 0; j < g.N; j++ {
		var ui float64 = 0
		max := g.X[0][j]
		min := g.X[0][j]
		for i := 0; i < g.M; i++ {
			ui += g.X[i][j]
			if g.X[i][j] > max {
				max = g.X[i][j]
			}
			if g.X[i][j] < min {
				min = g.X[i][j]
			}
		}
		ui /= float64(g.M)
		si := max - min
		for i := 0; i < g.M; i++ {
			g.X[i][j] = (g.X[i][j] - ui) / si
		}
	}
	fmt.Println("经过标准化之后的样本：")
	for i := 0; i < g.M; i++ {
		fmt.Printf("\t")
		fmt.Println(g.X[i])
	}

	// 梯度下降
	for t := 0; t < 50000; t++ {
		for j := 0; j < g.N + 1; j++ {
			g.Q[j] = g.Q[j] - g.A * linear_regression.J(j)
		}
	}
	fmt.Println("最终特征数组: ", g.Q)
	for i := 0; i < g.M; i++ {
		fmt.Printf("预测结果：%f，模型计算结果：%f\n", g.Y[i], linear_regression.H(i))
	}
}
