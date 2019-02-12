package common

type global struct {
	X [][]float64		// 训练样本
	Y []float64			// 预测结果
	Q []float64			// 标签
	A float64			// 学习速度
	M int				// 训练样本数
	N int				// 特征数（样本数+1）
}
var _global *global

func SetGlbVar(X [][]float64, Y []float64) *global {
	_global = new(global)
	_global.X = X
	_global.Y = Y
	_global.M = len(Y)
	_global.N = len(X[0])
	for t := 0; t < _global.N + 1; t++ {
		_global.Q = append(_global.Q, 1)
	}
	_global.A = 0.02
	return _global
}

func GetGlbVar() *global {
	if _global == nil {
		_global = new(global)
		_global.A = 0.02
	}
	return _global
}