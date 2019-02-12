package main

import (
	"common"
	"fmt"
	"log"
	"logistic_regression"
	"os"
	"tools"
)

const V = 0.002	// 学习速度
const A = 1e-5	// 精度
const MAX_LOOP = 1e7

func main() {
	// 读取训练数据集
	var err error
	var samples [][]float64
	var results []float64
	if samples, results, err = tools.LoadDataset("数据集/heart.csv", "target"); err != nil {
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
	g.X = tools.Normalize(g.X)
	fmt.Println("经过标准化之后的样本：")
	for i := 0; i < g.M; i++ {
		fmt.Println(g.X[i])
	}

	// 梯度下降
	var avgCost float64 = 1
	for t := 0; t < MAX_LOOP && avgCost > A; t++ {
		avgCost = 0
		for j := 0; j < g.N + 1; j++ {
			cost := logistic_regression.J(j)
			g.Q[j] = g.Q[j] - V * cost
			avgCost += cost
		}
		avgCost /= float64(g.N + 1)
		fmt.Println(avgCost)
	}
	fmt.Println("最终特征数组: ", g.Q)
	for i := 0; i < g.M; i++ {
		fmt.Printf("预测结果：%f，模型计算结果：%f\n", g.Y[i], logistic_regression.G(i))
	}
}
