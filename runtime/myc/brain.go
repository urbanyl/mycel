package myc

import (
	"math"
)

type edge struct {
	i, j int
	w    float64
}

// BrainView renders the liquid tissue as a live graph.
type BrainView struct {
	edges  []edge
	built  bool
	wmax   float64
	hist   [cells][160]float32
	head   int
	filled bool
}

func (bv *BrainView) build(c *Controller) {
	if bv.built {
		return
	}
	bv.wmax = 1e-9
	for i := 0; i < cells; i++ {
		for j := 0; j < cells; j++ {
			w := math.Abs(c.w[i][j])
			if w > bv.wmax {
				bv.wmax = w
			}
			if w > 0.16 {
				bv.edges = append(bv.edges, edge{i, j, c.w[i][j]})
			}
		}
	}
	bv.built = true
}

func (bv *BrainView) push(h *[cells]float64) {
	bv.head = (bv.head + 1) % len(bv.hist[0])
	for i := 0; i < cells; i++ {
		bv.hist[i][bv.head] = float32(h[i])
	}
}

// Draw paints the tissue panel into the framebuffer.
func (bv *BrainView) Draw(g *Gfx, c *Controller, x0, y0, w, h int) {
	bv.build(c)
	bv.push(&c.h)

	g.Rect(x0, y0, w, h, RGB(6, 11, 16))
	g.Rect(x0, y0, 1, h, RGB(22, 48, 60))

	pad := 18
	g.Text(x0+pad, y0+14, "LIQUID TISSUE", ColHud, 17)
	g.Text(x0+pad, y0+38, "24 cells  -  576 synapses", ColHudDim, 11)

	cx := float32(x0 + w/2)
	cy := float32(y0 + 210)
	r := float32(96)

	// ring of cells
	var px, py [cells]float32
	for i := 0; i < cells; i++ {
		a := -math.Pi/2 + 2*math.Pi*float64(i)/cells
		px[i] = cx + r*float32(math.Cos(a))
		py[i] = cy + r*float32(math.Sin(a))
	}

	for _, e := range bv.edges {
		k := math.Abs(e.w) / bv.wmax
		col := RGB(24, 74, 86)
		if e.w < 0 {
			col = RGB(62, 40, 96)
		}
		col = shade(col, 0.35+0.85*float32(k))
		g.Line2D(px[e.i], py[e.i], px[e.j], py[e.j], col, 1)
	}

	for i := 0; i < cells; i++ {
		a := math.Abs(c.h[i])
		rad := float32(3.2 + 7.5*a)
		col := ColJoint
		if c.h[i] < 0 {
			col = RGB(150, 110, 230)
		}
		col = shade(col, 0.45+0.9*float32(a))
		g.Disc(px[i], py[i], rad+1.6, RGB(8, 16, 22))
		g.Disc(px[i], py[i], rad, col)
	}

	// phase marker orbiting the ring
	a := -math.Pi/2 + c.phi
	mx := cx + (r+22)*float32(math.Cos(a))
	my := cy + (r+22)*float32(math.Sin(a))
	g.Disc(mx, my, 4.2, RGB(255, 184, 107))
	g.Disc(cx, cy, 2.5, RGB(40, 90, 104))
	g.Text(int(cx)-30, int(cy)+10, "CPG", ColHudDim, 10)

	// inputs
	iy := y0 + 330
	g.Text(x0+pad, iy, "SENSORY  x  /  14", ColHudDim, 11)
	bw := (w - 2*pad - 13*3) / inputs
	for i := 0; i < inputs; i++ {
		v := c.lastIn[i]
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		bar := int(math.Abs(v) * 26)
		bx := x0 + pad + i*(bw+3)
		g.Rect(bx, iy+54-13, bw, 26, RGB(12, 22, 30))
		col := RGB(60, 140, 160)
		if v < 0 {
			col = RGB(110, 80, 170)
		}
		if v >= 0 {
			g.Rect(bx, iy+54-13+13-bar/2, bw, bar/2+1, col)
		} else {
			g.Rect(bx, iy+54-13+13, bw, bar/2+1, col)
		}
	}

	// outputs
	oy := y0 + 412
	g.Text(x0+pad, oy, "MOTOR  y  /  8", ColHudDim, 11)
	obw := (w - 2*pad - 7*6) / outputs
	names := [8]string{"dxL", "liftL", "dxR", "liftR", "lean", "pushL", "pushR", "-"}
	for i := 0; i < outputs; i++ {
		v := c.lastOut[i]
		bar := int(math.Abs(v) * 22)
		bx := x0 + pad + i*(obw+6)
		by := oy + 34
		g.Rect(bx, by, obw, 24, RGB(12, 22, 30))
		col := RGB(70, 190, 200)
		if v < 0 {
			col = RGB(140, 100, 200)
		}
		g.Rect(bx, by+12-bar/2, obw, bar/2+1, col)
		g.Text(bx-2, by+30, names[i], ColHudDim, 9)
	}

	// activity raster
	ry := y0 + 530
	g.Text(x0+pad, ry-16, "ACTIVITY  /  last 2.7 s", ColHudDim, 11)
	rowH := 8
	cols := len(bv.hist[0])
	start := bv.head
	for i := 0; i < cells; i++ {
		for k := 0; k < 150; k++ {
			idx := (start + 1 + k*cols/150) % cols
			v := float32(bv.hist[i][idx])
			a := math.Abs(float64(v))
			if a < 0.04 {
				continue
			}
			col := shade(ColJoint, float32(0.3+a))
			if v < 0 {
				col = shade(RGB(150, 110, 230), float32(0.3+a))
			}
			g.Rect(x0+pad+k*2, ry+i*rowH, 2, rowH-2, col)
		}
	}

	g.Text(x0+pad, y0+h-24, "W h + U x + b  ->  tanh  ->  h", RGB(40, 70, 84), 10)
}
