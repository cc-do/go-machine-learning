package tests

import (
	"testing"
	"fmt"
	"neural_networks"
)

func TestNeuralNetworks(t *testing.T) {
	machine := new(neural_networks.NeuNwk)
	result := machine.GenBasicNeuNwk(5, 7)
	fmt.Println(result)
}