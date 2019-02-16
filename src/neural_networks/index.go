package neural_networks

import (
	"tools"
	"math"
	"linear_regression"
	"strings"
	"fmt"
)

type Neuron struct {
	Q []float64		// 权重参数
	Siblings *Neuron
}

func (neu *Neuron) activate(inputs []float64) float64 {
	// 将X0设为1，与Q0相乘形成截距
	inputs = append([]float64{1}, inputs...)
	z := tools.DotMulti(neu.Q, inputs)
	// 使用sigmord函数作为激活函数
	return 1/(1 + math.Exp(-z))
}

type NeuNwk struct {
	linear_regression.LinReg
	K []float64		// 基数集合
	Q []*Neuron		// 所有神经元组（接受输入的神经元）
	L int			// 层数
}

func (nn *NeuNwk) SetDataset(samples [][]float64, results []float64) {
	nn.X = samples
	nn.Y = results
	nn.M = len(nn.Y)
	nn.N = len(nn.X[0])
	// 计算输出空间的离散样本数
	nn.K = tools.Cardinality(nn.Y)
	// 生成神经网络（基本神经网络）
	nn.GenNeuralNetworks("basic")
	// 标准化样本数据
	nn.X = tools.Normalize(nn.X)
	fmt.Println("经过标准化之后的样本：")
	for i := 0; i < nn.M; i++ {
		fmt.Println(nn.X[i])
	}
}

func (nn *NeuNwk) J(j int) float64 {
	return 0
}

func (nn *NeuNwk) Learn(maxLoop int, accuracy float64, velocity float64) {

}

func (nn *NeuNwk) GenNeuralNetworks(nkTyp string) {
	switch strings.ToLower(nkTyp) {
	case "basic": nn.Q = nn.GenBasicNeuNwk(nn.N, len(nn.K))
	}
}

func (nn *NeuNwk) GenBasicNeuNwk(inputs int, outputs int) []*Neuron {
	acc := 1
	if inputs > outputs {
		acc = -1
	}
	var ret []*Neuron
	for i := inputs; i != outputs; i += acc {
		neuron := nn.NewNeuron(i + 1)
		ret = append(ret, neuron)
		for j := 0; j < i; j++ {
			neuron.Siblings = nn.NewNeuron(i + 1)
			neuron = neuron.Siblings
		}
	}
	return ret
}

func (nn *NeuNwk) NewNeuron(numQ int) *Neuron {
	ret := new(Neuron)
	for i := 0; i < numQ; i++ {
		ret.Q = append(ret.Q, 1)
	}
	return ret
}