package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

// 全局变量
var X [][]float64		// 训练样本
var Y []float64			// 预测结果
var Q []float64			// 标签
const a float64 = 0.02	// 学习速度
var m = 0				// 训练样本数
var n = 0				// 特征数（样本数+1）

// Global functions
func loadDataset(predictColumn string) {
	// 打开数据集
	file, err := os.Open("数据集/heart.csv")
	defer file.Close()
	if err != nil {
		log.Fatal(err)
		return
	}

	// 从数据集中读取
	reader := csv.NewReader(file)
	predictColIdx := -1
	for i := 0;; i++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
			return
		}
		// 跳过标题行
		if i == 0 {
			n = len(record)
			if predictColumn != "" {
				n--
			}
			// 记录预测标题的列索引号
			for j := 0; j < n; j++ {
				if record[j] == predictColumn {
					predictColIdx = j
					break
				}
			}
			continue
		}

		m++
		// 如果存在预测结果，则总样本列应该加上这列
		colNum := n
		if predictColIdx != -1 {
			colNum++
		}
		// 将string类型的样本数据转化成float64
		temp := make([]float64, colNum)
		for j := 0; j < colNum; j++ {
			temp[j], err = strconv.ParseFloat(record[j], 64)
			if err != nil {
				log.Fatal(fmt.Printf("数据集中包含非数字格式：%s\n", record[j]))
				return
			}
		}

		// 如果有预测结果列，则temp数组中的该列项归为预测结果，剩下的都是训练数据
		if predictColIdx != -1 {
			Y = append(Y, temp[predictColIdx])
			temp = append(temp[:predictColIdx], temp[predictColIdx + 1:]...)
			X = append(X, temp)
		} else {
			X = append(X, temp)
		}
	}

	fmt.Println(X)
	fmt.Println(Y)
	fmt.Println(m)
	fmt.Println(n)
}

// Sigmoid函数
func g(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Hypothesis function
func h(i int) float64 {
	var sum = Q[0]
	for j := 1; j < n; j++ {
		sum += Q[j] * X[i][j-1]
	}
	return g(sum)
}

// Cost function
func J(j int) float64 {
	var sum float64 = 0
	for i := 0; i < m; i++ {
		// X0 = 1 otherwise use sample
		var xij float64 = 1
		if j != 0 {
			xij = X[i][j-1]
		}
		sum += (h(i) - Y[i]) * xij
	}
	return sum / float64(m)
}

// 打印进度条
func printProcs(proc int) {
	fmt.Printf("\n[%d%%]", proc)
	for i := 0; i < 100; i++ {
		if i < proc {
			fmt.Print(">")
		} else {
			fmt.Print("=")
		}
	}
	fmt.Println()
}

func main() {
	// 读取训练数据集
	loadDataset("chol")
	if len(X) == 0 || len(Y) == 0 {
		os.Exit(-1)
	}
	if len(X) != len(Y) {
		log.Fatal("samples' size should be same to results' size")
		os.Exit(2)
	}
	m = len(Y)
	n = len(X[0]) + 1
	for t := 0; t < n; t++ {
		Q = append(Q, 1)
	}

	// Normalize samples
	for j := 0; j < n-1; j++ {
		var ui float64 = 0
		max := X[0][j]
		min := X[0][j]
		for i := 0; i < m; i++ {
			ui += X[i][j]
			if X[i][j] > max {
				max = X[i][j]
			}
			if X[i][j] < min {
				min = X[i][j]
			}
		}
		ui /= float64(m)
		si := max - min
		for i := 0; i < m; i++ {
			X[i][j] = (X[i][j] - ui) / si
		}
	}
	fmt.Println("After normalized samples: ")
	for i := 0; i < m; i++ {
		fmt.Printf("\t")
		fmt.Println(X[i])
	}

	// Gradient descent
	printProcs(0)
	for t := 0; t < 50000; t++ {
		for j := 0; j < n; j++ {
			Q[j] = Q[j] - a * J(j)
		}
		printProcs(t/500)
	}
	fmt.Println("最终特征数组: ", Q)
	for i := 0; i < m; i++ {
		fmt.Printf("Result: %f, Calculte: %f\n", Y[i], h(i))
	}
}
