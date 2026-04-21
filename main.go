// Copyright 2026 The Hamiltonian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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

// Hadamard computes the hadamard product of two tensors
func Hadamard(k tf64.Continuation, node int, a, b *tf64.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	c := tf64.NewV(a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j*b.X[i%length])
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * b.X[i%length]
		b.D[i%length] += j * a.X[i]
	}
	return false
}

const (
	// U is the size of the universe
	U = 1.0e26
	// V is the speed of light
	V = 299792458.0
)

// LearnEmbedding learns the embeddings
func LearnEmbedding(inputs Matrix[float64], width, iterations int) (float64, []float64, [][]float64) {
	const Eta = 1e-3
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	others.Add("x", inputs.Cols, inputs.Rows)
	x := others.ByName["x"]
	for row := range inputs.Rows {
		for _, value := range inputs.Data[row*inputs.Cols : row*inputs.Cols+inputs.Cols] {
			x.X = append(x.X, value)
		}
	}
	others.Add("c", 1, 1)
	others.ByName["c"].X = append(others.ByName["c"].X, V)

	set := tf64.NewSet()
	set.Add("i", width, inputs.Rows)
	set.Add("g", 1, 1)
	//set.Add("l", 1, 1)

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
	//set.ByName["g"].X[0] = 1e-11
	//set.ByName["l"].X[0] = U

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	hadamard := tf64.B(Hadamard)
	//c := tf64.Inv(hadamard(set.Get("l"), set.Get("g")))
	//c := tf64.Inv(others.Get("c"))
	sa := tf64.Mul(tf64.Dropout(tf64.Square( /*hadamard(*/ set.Get("i") /*, c)*/), dropout), hadamard(others.Get("x"), set.Get("g")))
	loss := tf64.Avg(tf64.Quadratic(hadamard(others.Get("x"), set.Get("g")), sa))

	var l float64
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
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return 0.0, nil, nil
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
	}
	fmt.Println(l)

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
	return others.ByName["c"].X[0], set.ByName["g"].X, outputs
}

// LearnG learns g
func LearnG(inputs Matrix[float64], width, iterations int) (float64, []float64, [][]float64) {
	const Eta = 1e-3
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	others.Add("x", inputs.Cols, inputs.Rows)
	x := others.ByName["x"]
	for row := range inputs.Rows {
		for _, value := range inputs.Data[row*inputs.Cols : row*inputs.Cols+inputs.Cols] {
			x.X = append(x.X, value)
		}
	}
	others.Add("c", 1, 1)
	others.ByName["c"].X = append(others.ByName["c"].X, V)

	set := tf64.NewSet()
	set.Add("i", width, inputs.Rows)
	set.Add("g", inputs.Cols, inputs.Rows)
	//set.Add("l", 1, 1)

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
	for i := range set.ByName["g"].X {
		set.ByName["g"].X[i] = 1e-11
	}
	//set.ByName["l"].X[0] = U

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	hadamard := tf64.B(Hadamard)
	//c := tf64.Inv(hadamard(set.Get("l"), set.Get("g")))
	//c := tf64.Inv(others.Get("c"))
	sa := tf64.Mul(tf64.Dropout(tf64.Square( /*hadamard(*/ set.Get("i") /*, c)*/), dropout), hadamard(others.Get("x"), set.Get("g")))
	loss := tf64.Avg(tf64.Quadratic(tf64.Mul(hadamard(others.Get("x"), set.Get("g")), tf64.Dropout(tf64.Square( /*hadamard(*/ set.Get("i") /*, c)*/), dropout)), sa))

	var l float64
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
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return 0.0, nil, nil
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
	}
	fmt.Println(l)

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
	return others.ByName["c"].X[0], set.ByName["g"].X, outputs
}

var (
	// FlagS s mode
	FlagS = flag.Bool("s", false, "s mode")
	// FlagEpochs number of epochs
	FlagEpochs = flag.Int("e", 1, "number of epochs")
)

// SMode s mode
func SMode(epochs int, iterate func(inputs Matrix[float64], width, iterations int) (float64, []float64, [][]float64)) {
	rng := rand.New(rand.NewSource(1))
	g := NewMatrix[float64](3, 33)
	for range g.Rows {
		for range g.Cols {
			g.Data = append(g.Data, rng.Float64())
		}
	}
	getadj := func() Matrix[float64] {
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
		return gadj
	}
	gadj := getadj()
	fmt.Println(gadj.Data)
	images := &gif.GIF{}
	var palette = []color.Color{}
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
	delay := make([][]chan float64, g.Rows)
	for i := range delay {
		delay[i] = make([]chan float64, g.Cols)
		for ii := range delay[i] {
			delay[i][ii] = make(chan float64, 2)
		}
	}
	gs := make(plotter.XYs, 0, 8)
	gavg := make(plotter.XYs, 0, 8)
	var gshist plotter.Values
	var chist plotter.Values
	for epoch := range epochs {
		fmt.Println(epoch)
		l, G, outputs := iterate(gadj, 3, 512)
		for i := range outputs {
			type R struct {
				R float64
				I int
			}
			r := make([]R, g.Rows)
			for ii := range r {
				r[ii].R = gadj.Data[i*gadj.Rows+ii]
				r[ii].I = ii
			}
			sort.Slice(r, func(i, j int) bool {
				return r[i].R < r[j].R
			})
			//split := r[len(r)/2]
			for ii := range outputs[i] {
				v := outputs[i][ii]
				/*if v > split.R {
					select {
					case vv := <-delay[i][ii]:
						delay[i][ii] <- v
						g.Data[i*g.Cols+ii] += vv
					default:
						delay[i][ii] <- v
					}
				} else {*/
				g.Data[i*g.Cols+ii] += v
				//}
			}
		}
		if epoch < 1024 {
			image := image.NewPaletted(image.Rect(0, 0, 1024, 1024), palette)
			type Offset struct {
				X int
				Y int
				A int
				B int
			}
			offsets := []Offset{{0, 0, 0, 1}, {512, 0, 0, 2}, {0, 512, 1, 2}}
			for _, offset := range offsets {
				minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
				for i := range g.Rows {
					x, y := g.Data[i*g.Cols+offset.A], g.Data[i*g.Cols+offset.B]
					if x < minX {
						minX = x
					}
					if x > maxX {
						maxX = x
					}
					if y < minY {
						minY = y
					}
					if y > maxY {
						maxY = y
					}
				}
				for i := range g.Rows {
					xx, yy := g.Data[i*g.Cols+offset.A], g.Data[i*g.Cols+offset.B]
					x := 500*(xx-minX)/(maxX-minX) + 6
					y := 500*(yy-minY)/(maxY-minY) + 6
					image.Set(offset.X+int(x), offset.Y+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
			images.Image = append(images.Image, image)
			images.Delay = append(images.Delay, 10)
		}
		gadj = getadj()
		avg := 0.0
		for _, value := range G {
			avg += value
		}
		avg /= float64(len(G))
		stddev := 0.0
		for _, value := range G {
			diff := value - avg
			stddev = diff * diff
		}
		stddev /= float64(len(G))
		stddev = math.Sqrt(stddev)
		fmt.Println("c", l, "G", avg, stddev)
		gavg = append(gavg, plotter.XY{X: float64(epoch), Y: float64(avg)})
		for _, G := range G {
			gs = append(gs, plotter.XY{X: float64(epoch), Y: float64(G)})
			gshist = append(gshist, float64(G))
		}
		chist = append(chist, float64(l))
		//gg = G
	}
	out, err := os.Create("verse.gif")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	err = gif.EncodeAll(out, images)
	if err != nil {
		panic(err)
	}
	fmt.Println(g.Data)

	p := plot.New()

	p.Title.Text = "G vs time"
	p.X.Label.Text = "time"
	p.Y.Label.Text = "G"

	scatter, err := plotter.NewScatter(gs)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "G.png")
	if err != nil {
		panic(err)
	}

	{
		p := plot.New()

		p.Title.Text = "G vs time"
		p.X.Label.Text = "time"
		p.Y.Label.Text = "G"

		scatter, err := plotter.NewScatter(gavg)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "Gavg.png")
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()
		p.Title.Text = "G"

		hist, err := plotter.NewHist(gshist, 256)
		if err != nil {
			panic(err)
		}
		max, index := 0.0, 0
		for i, bin := range hist.Bins {
			if bin.Weight > max {
				max, index = bin.Weight, i
			}
		}
		{
			min, max := hist.Bins[index].Min, hist.Bins[index].Max
			fmt.Println("min max", min, max)
			fmt.Println("min^2 max^2", min*min, max*max)
			fmt.Println("min/c max/c", min/V, max/V)
		}
		p.Add(hist)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "Ghist.png")
		if err != nil {
			panic(err)
		}

		sort.Slice(hist.Bins, func(i, j int) bool {
			return hist.Bins[i].Weight < hist.Bins[j].Weight
		})
		for i := range hist.Bins {
			fmt.Println(hist.Bins[i])
		}

		fmt.Println()
		histogram := make(map[int]int)
		for _, value := range gshist {
			exp := int(math.Floor(math.Log10(math.Abs(value))))
			count := histogram[exp]
			count++
			histogram[exp] = count
		}
		type Count struct {
			Count int
			Exp   int
		}
		counts := make([]Count, 0, len(histogram))
		for key, value := range histogram {
			counts = append(counts, Count{
				Count: value,
				Exp:   key,
			})
		}
		sort.Slice(counts, func(i, j int) bool {
			return counts[i].Count < counts[j].Count
		})
		for _, count := range counts {
			fmt.Println(count.Exp, ":", count.Count)
		}
	}

	{
		fmt.Println()
		histogram := make(map[int]int)
		for _, value := range chist {
			exp := int(math.Floor(math.Log10(math.Abs(value))))
			count := histogram[exp]
			count++
			histogram[exp] = count
		}
		type Count struct {
			Count int
			Exp   int
		}
		counts := make([]Count, 0, len(histogram))
		for key, value := range histogram {
			counts = append(counts, Count{
				Count: value,
				Exp:   key,
			})
		}
		sort.Slice(counts, func(i, j int) bool {
			return counts[i].Count < counts[j].Count
		})
		for _, count := range counts {
			fmt.Println(count.Exp, ":", count.Count)
		}
	}
}

func main() {
	flag.Parse()

	if *FlagS {
		SMode(*FlagEpochs*1024, LearnEmbedding)
		return
	}

	SMode(*FlagEpochs*1024, LearnG)
}
