// Copyright 2026 The Hamiltonian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// LearnEmbedding learns the embeddings
func LearnEmbedding(inputs Matrix[float64], width, iterations int) [][]float64 {
	const Eta = 1e-3
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	others.Add("x", inputs.Cols, inputs.Rows)
	x := others.ByName["x"]
	for row := range inputs.Rows {
		for _, value := range inputs.Data[row*inputs.Cols : row*inputs.Cols+inputs.Cols] {
			x.X = append(x.X, value*1e-3)
		}
	}

	set := tf64.NewSet()
	set.Add("i", width, inputs.Rows)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range iterations {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				_ = mhat
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				/*if rng.Float64() > .01 {
					w.X[ii] -= .05 * g
				} else {
					w.X[ii] += .05 * g
				}*/
			}
		}
		fmt.Println(l)
	}

	/*meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*size:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})*/
	I := set.ByName["i"]
	outputs := make([][]float64, inputs.Rows)
	for i := range outputs {
		outputs[i] = I.X[i*width : (i+1)*width]
	}
	return outputs
}

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
	outputs := LearnEmbedding(gadj, 3, 256)
	fmt.Println(outputs)
}
