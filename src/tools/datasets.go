package tools

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
)

func LoadDataset(csvFile string, predictColumn string) ([][]float64, []float64, error) {
	var X [][]float64
	var Y []float64

	// 打开数据集
	file, err := os.Open(csvFile)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
		return nil, nil, err
	}

	// 从数据集中读取
	reader := csv.NewReader(file)
	predictColIdx := -1
	featureNum := 0
	for i := 0;; i++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
			return nil, nil, err
		}
		// 跳过标题行
		if i == 0 {
			featureNum = len(record)
			if predictColumn != "" {
				featureNum--
			}
			// 记录预测标题的列索引号
			for j := 0; j < featureNum; j++ {
				if record[j] == predictColumn {
					predictColIdx = j
					break
				}
			}
			continue
		}

		// 如果存在预测结果，则总样本列应该加上这列
		colNum := featureNum
		if predictColIdx != -1 {
			colNum++
		}
		// 将string类型的样本数据转化成float64
		temp := make([]float64, colNum)
		for j := 0; j < colNum; j++ {
			temp[j], err = strconv.ParseFloat(record[j], 64)
			if err != nil {
				log.Fatal(fmt.Printf("数据集中包含非数字格式：%s\n", record[j]))
				return nil, nil, err
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
	return X, Y, nil
}