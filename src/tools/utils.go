package tools

import "fmt"

// 打印进度条
func PrintProcs(proc int) {
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