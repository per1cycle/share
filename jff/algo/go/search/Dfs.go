package main

import "fmt"

var pos[10] int;
var stat[10] int;

/**
 * recursive visit the position
 * 
 */
func solution(n int, total_len int) int {
	if (n == total_len) {
		fmt.Println(total_len)
		return 0;
	}

	for i := 0; i < total_len; i ++ {
		if (pos[i]) {
			continue
		}
		else {
			pos[n] = i
			stat[i] = 1
			solution(n + 1, total_len)
			stat[i] = 0
		}
	}
	return 1
}

func main() {
	var n int

	fmt.Scan(&n)

	solution(0, n)
}
