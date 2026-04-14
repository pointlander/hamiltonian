// Copyright 2026 The Hamiltonian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	rng := rand.New(rand.NewSource(1))
	g := NewMatrix[float64](3, 33)
	for range 33 {
		for range 3 {
			g.Data = append(g.Data, rng.Float64())
		}
	}
	gadj := NewMatrix[float64](g.Rows, g.Rows)
	for i := range g.Rows {
		for j := range g.Rows {
			sum := 0.0
			for k := range g.Cols {
				diff := g.Data[i*g.Cols+k] - g.Data[j*g.Cols+k]
				if diff < 0 {
					diff = -diff
				}
				sum += diff * diff
			}
			distance := math.Sqrt(sum)
			if distance == 0 {
				gadj.Data = append(gadj.Data, 0)
				continue
			}
			gadj.Data = append(gadj.Data, 1/distance)
		}
	}
	fmt.Println(gadj.Data)
}
