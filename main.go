package main

import (
	"log"
	"os"
	"tools"
	"softmax_regression"
)

const V = 0.0002// 学习速度
const A = 1e-7	// 精度
const MAX_LOOP = 1e5

func main() {
	// 读取训练数据集
	var err error
	var samples [][]float64
	var results []float64
	if samples, results, err = tools.LoadDataset("数据集/heart.csv", "thal"); err != nil {
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

	machine := new(softmax_regression.SmxReg)
	machine.SetDataset(samples, results)

	// 梯度下降
	machine.Learn(MAX_LOOP, A, V)

	// 测试
	machine.Test()
}
