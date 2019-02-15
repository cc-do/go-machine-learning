package tools

type Method interface {
	SetDataset(samples [][]float64, results []float64)
	Learn(maxLoop int, accuracy float64, velocity float64)
	Test()
}