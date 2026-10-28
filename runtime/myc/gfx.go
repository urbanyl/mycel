package myc

import "math"

// ---------------------------------------------------------------------------
// linear algebra
// ---------------------------------------------------------------------------

type V3 struct{ X, Y, Z float32 }

func V(x, y, z float32) V3 { return V3{x, y, z} }

func (a V3) Add(b V3) V3      { return V3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a V3) Sub(b V3) V3      { return V3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a V3) Mul(s float32) V3 { return V3{a.X * s, a.Y * s, a.Z * s} }
func (a V3) Dot(b V3) float32 { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a V3) Cross(b V3) V3 {
	return V3{a.Y*b.Z - a.Z*b.Y, a.Z*b.X - a.X*b.Z, a.X*b.Y - a.Y*b.X}
}
func (a V3) Len() float32 { return float32(math.Sqrt(float64(a.Dot(a)))) }
func (a V3) Unit() V3 {
	l := a.Len()
	if l < 1e-9 {
		return V3{}
	}
	return a.Mul(1 / l)
}

type M4 [16]float32

func ident() M4 {
	return M4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

func (m M4) Mul(n M4) M4 {
	var r M4
	for c := 0; c < 4; c++ {
		for row := 0; row < 4; row++ {
			var s float32
			for k := 0; k < 4; k++ {
				s += m[k*4+row] * n[c*4+k]
			}
			r[c*4+row] = s
		}
	}
	return r
}

func (m M4) Apply(p V3) (V3, float32) {
	x := m[0]*p.X + m[4]*p.Y + m[8]*p.Z + m[12]
	y := m[1]*p.X + m[5]*p.Y + m[9]*p.Z + m[13]
	z := m[2]*p.X + m[6]*p.Y + m[10]*p.Z + m[14]
	w := m[3]*p.X + m[7]*p.Y + m[11]*p.Z + m[15]
	return V3{x, y, z}, w
}

func lookAt(eye, target, up V3) M4 {
	f := target.Sub(eye).Unit()
	s := f.Cross(up).Unit()
	u := s.Cross(f)
	return M4{
		s.X, u.X, -f.X, 0,
		s.Y, u.Y, -f.Y, 0,
		s.Z, u.Z, -f.Z, 0,
		-s.Dot(eye), -u.Dot(eye), f.Dot(eye), 1,
	}
}

func perspective(fovy, aspect, near, far float32) M4 {
	f := 1 / float32(math.Tan(float64(fovy)/2))
	nf := 1 / (near - far)
	return M4{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, (far + near) * nf, -1,
		0, 0, 2 * far * near * nf, 0,
	}
}

// Camera is a pinhole camera with a cached view-projection matrix.
type Camera struct {
	Pos, Target, Up V3
	Fov             float32 // radians
	Near, Far       float32
	VX, VY, VW, VH  int     // viewport inside the framebuffer
	m               M4
}

func NewCamera(pos, target V3, fovDeg, near, far float32) *Camera {
	c := &Camera{Pos: pos, Target: target, Up: V3{0, 1, 0},
		Fov: fovDeg * math.Pi / 180, Near: near, Far: far}
	c.m = perspective(c.Fov, 1, near, far).Mul(lookAt(pos, target, c.Up))
	return c
}

func (c *Camera) Update(w, h int) {
	c.VX, c.VY, c.VW, c.VH = 0, 0, w, h
	c.m = perspective(c.Fov, float32(w)/float32(h), c.Near, c.Far).
		Mul(lookAt(c.Pos, c.Target, c.Up))
}

// Viewport restricts the camera to a sub-rectangle of the framebuffer.
func (c *Camera) Viewport(x, y, w, h int) {
	c.VX, c.VY, c.VW, c.VH = x, y, w, h
	c.m = perspective(c.Fov, float32(w)/float32(h), c.Near, c.Far).
		Mul(lookAt(c.Pos, c.Target, c.Up))
}

// project returns screen coordinates and a depth for the z-buffer.
func (c *Camera) project(p V3, w, h int) (float32, float32, float32, bool) {
	v, cw := c.m.Apply(p)
	if cw <= 1e-4 {
		return 0, 0, 0, false
	}
	inv := 1 / cw
	ndcX := v.X * inv
	ndcY := v.Y * inv
	ndcZ := v.Z * inv
	if ndcZ < -1 || ndcZ > 1 {
		return 0, 0, 0, false
	}
	vw, vh := c.VW, c.VH
	if vw == 0 || vh == 0 {
		vw, vh = w, h
	}
	sx := float32(c.VX) + (ndcX*0.5+0.5)*float32(vw)
	sy := float32(c.VY) + (0.5-ndcY*0.5)*float32(vh)
	return sx, sy, ndcZ, true
}

// ---------------------------------------------------------------------------
// framebuffer
// ---------------------------------------------------------------------------

const (
	Black = 0x00000000
)

func RGB(r, g, b uint8) uint32 {
	return uint32(r)<<16 | uint32(g)<<8 | uint32(b)
}

func mix(a, b uint32, t float32) uint32 {
	if t <= 0 {
		return a
	}
	if t >= 1 {
		return b
	}
	ar, ag, ab := float32(a>>16&255), float32(a>>8&255), float32(a&255)
	br, bg, bb := float32(b>>16&255), float32(b>>8&255), float32(b&255)
	r := uint32(ar + (br-ar)*t)
	g := uint32(ag + (bg-ag)*t)
	bl := uint32(ab + (bb-ab)*t)
	return r<<16 | g<<8 | bl
}

func shade(c uint32, k float32) uint32 {
	if k < 0 {
		k = 0
	}
	if k > 1.6 {
		k = 1.6
	}
	r := uint32(math.Min(255, float64(float32(c>>16&255)*k)))
	g := uint32(math.Min(255, float64(float32(c>>8&255)*k)))
	b := uint32(math.Min(255, float64(float32(c&255)*k)))
	return r<<16 | g<<8 | b
}

// Gfx is a direct-mapped framebuffer with a depth buffer.
type Gfx struct {
	W, H int
	Pix  []uint32
	Z    []float32
}

func (g *Gfx) Clear(c uint32) {
	for i := range g.Pix {
		g.Pix[i] = c
	}
	for i := range g.Z {
		g.Z[i] = 1e9
	}
}

func (g *Gfx) plot(x, y int, c uint32) {
	if x < 0 || y < 0 || x >= g.W || y >= g.H {
		return
	}
	g.Pix[y*g.W+x] = c
}

func (g *Gfx) hline(y, x0, x1 int, c uint32) {
	if y < 0 || y >= g.H {
		return
	}
	if x0 > x1 {
		x0, x1 = x1, x0
	}
	if x0 < 0 {
		x0 = 0
	}
	if x1 >= g.W {
		x1 = g.W - 1
	}
	row := g.Pix[y*g.W : y*g.W+g.W]
	for x := x0; x <= x1; x++ {
		row[x] = c
	}
}

// Line2D draws a thick line in screen space, no depth test.
func (g *Gfx) Line2D(x0, y0, x1, y1 float32, c uint32, width int) {
	dx := x1 - x0
	dy := y1 - y0
	n := int(math.Abs(float64(dx)))
	if int(math.Abs(float64(dy))) > n {
		n = int(math.Abs(float64(dy)))
	}
	if n == 0 {
		n = 1
	}
	hw := float32(width-1) * 0.5
	for i := 0; i <= n; i++ {
		t := float32(i) / float32(n)
		x := x0 + dx*t
		y := y0 + dy*t
		if width <= 1 {
			g.plot(int(x+0.5), int(y+0.5), c)
			continue
		}
		for oy := -hw; oy <= hw; oy += 1 {
			for ox := -hw; ox <= hw; ox += 1 {
				g.plot(int(x+ox+0.5), int(y+oy+0.5), c)
			}
		}
	}
}

func (g *Gfx) Disc(cx, cy, r float32, c uint32) {
	r2 := r * r
	x0 := int(cx - r - 1)
	x1 := int(cx + r + 1)
	y0 := int(cy - r - 1)
	y1 := int(cy + r + 1)
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			dx := float32(x) - cx
			dy := float32(y) - cy
			if dx*dx+dy*dy <= r2 {
				g.plot(x, y, c)
			}
		}
	}
}

func (g *Gfx) Rect(x, y, w, h int, c uint32) {
	for j := 0; j < h; j++ {
		g.hline(y+j, x, x+w-1, c)
	}
}

// Text is rendered by GDI on top of the framebuffer (see window.go).
type textItem struct {
	x, y int
	s    string
	c    uint32
	size int
}

// ---------------------------------------------------------------------------
// 3D rasteriser
// ---------------------------------------------------------------------------

type light struct{ Dir V3 }

var defaultLight = V3{-0.45, -0.75, 0.48}.Unit()

func (g *Gfx) tri3(ax, ay, az float32, bx, by, bz float32, cx, cy, cz float32,
	a V3, b V3, c V3, col uint32, cam *Camera) {

	n := b.Sub(a).Cross(c.Sub(a)).Unit()
	k := 0.34 + 0.78*float32(math.Abs(float64(n.Dot(defaultLight))))
	cc := shade(col, k)

	minX := int(math.Floor(float64(min3(ax, bx, cx))))
	maxX := int(math.Ceil(float64(max3(ax, bx, cx))))
	minY := int(math.Floor(float64(min3(ay, by, cy))))
	maxY := int(math.Ceil(float64(max3(ay, by, cy))))
	if minX < 0 {
		minX = 0
	}
	if minY < 0 {
		minY = 0
	}
	if maxX >= g.W {
		maxX = g.W - 1
	}
	if maxY >= g.H {
		maxY = g.H - 1
	}
	if minX > maxX || minY > maxY {
		return
	}

	area := (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
	if area == 0 {
		return
	}
	inv := 1 / area

	for y := minY; y <= maxY; y++ {
		py := float32(y) + 0.5
		row := y * g.W
		for x := minX; x <= maxX; x++ {
			px := float32(x) + 0.5
			w0 := ((bx-px)*(cy-py) - (by-py)*(cx-px)) * inv
			if w0 < 0 {
				continue
			}
			w1 := ((cx-px)*(ay-py) - (cy-py)*(ax-px)) * inv
			if w1 < 0 {
				continue
			}
			w2 := 1 - w0 - w1
			if w2 < 0 {
				continue
			}
			z := w0*az + w1*bz + w2*cz
			if z >= g.Z[row+x] {
				continue
			}
			g.Z[row+x] = z
			g.Pix[row+x] = cc
		}
	}
}

func min3(a, b, c float32) float32 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}

func max3(a, b, c float32) float32 {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	return m
}

// Tri projects and rasterises one world-space triangle.
func (g *Gfx) Tri(a, b, c V3, col uint32, cam *Camera) {
	ax, ay, az, ok0 := cam.project(a, g.W, g.H)
	if !ok0 {
		return
	}
	bx, by, bz, ok1 := cam.project(b, g.W, g.H)
	if !ok1 {
		return
	}
	cx, cy, cz, ok2 := cam.project(c, g.W, g.H)
	if !ok2 {
		return
	}
	g.tri3(ax, ay, az, bx, by, bz, cx, cy, cz, a, b, c, col, cam)
}

// Quad emits two triangles.
func (g *Gfx) Quad(a, b, c, d V3, col uint32, cam *Camera) {
	g.Tri(a, b, c, col, cam)
	g.Tri(a, c, d, col, cam)
}

// Beam draws a rectangular beam between two points, extruded along Z.
func (g *Gfx) Beam(a, b V3, width, depth float32, col uint32, cam *Camera) {
	d := b.Sub(a)
	l := float32(math.Sqrt(float64(d.X*d.X + d.Y*d.Y)))
	if l < 1e-6 {
		return
	}
	nx := -d.Y / l * width * 0.5
	ny := d.X / l * width * 0.5
	hz := depth * 0.5

	p := [4]V3{
		{a.X + nx, a.Y + ny, a.Z - hz},
		{b.X + nx, b.Y + ny, b.Z - hz},
		{b.X - nx, b.Y - ny, b.Z - hz},
		{a.X - nx, a.Y - ny, a.Z - hz},
	}
	q := [4]V3{
		{a.X + nx, a.Y + ny, a.Z + hz},
		{b.X + nx, b.Y + ny, b.Z + hz},
		{b.X - nx, b.Y - ny, b.Z + hz},
		{a.X - nx, a.Y - ny, a.Z + hz},
	}
	g.Quad(p[0], p[1], p[2], p[3], col, cam)
	g.Quad(q[3], q[2], q[1], q[0], shade(col, 0.82), cam)
	g.Quad(p[0], q[0], q[1], p[1], shade(col, 1.12), cam)
	g.Quad(p[1], q[1], q[2], p[2], shade(col, 0.95), cam)
	g.Quad(p[3], q[3], q[2], p[2], shade(col, 0.9), cam)
	g.Quad(p[0], q[0], q[3], p[3], shade(col, 0.9), cam)
}

// Box draws an axis-aligned box.
func (g *Gfx) Box(c V3, sx, sy, sz float32, col uint32, cam *Camera) {
	x0, x1 := c.X-sx/2, c.X+sx/2
	y0, y1 := c.Y-sy/2, c.Y+sy/2
	z0, z1 := c.Z-sz/2, c.Z+sz/2
	a := V3{x0, y1, z1}
	b := V3{x1, y1, z1}
	d := V3{x1, y1, z0}
	e := V3{x0, y1, z0}
	g.Quad(a, b, d, e, shade(col, 1.2), cam)
	g.Quad(e, d, b, a, shade(col, 0.55), cam)
	g.Quad(V3{x0, y0, z1}, V3{x1, y0, z1}, V3{x1, y1, z1}, V3{x0, y1, z1}, col, cam)
	g.Quad(V3{x0, y0, z0}, V3{x0, y0, z1}, V3{x0, y1, z1}, V3{x0, y1, z0}, shade(col, 0.85), cam)
	g.Quad(V3{x1, y0, z0}, V3{x1, y0, z1}, V3{x1, y1, z1}, V3{x1, y1, z0}, shade(col, 0.85), cam)
}

// Grid draws a perspective ground grid with fade.
func (g *Gfx) Grid(cam *Camera, half int, step float32, col uint32) {
	for i := -half; i <= half; i++ {
		x := float32(i) * step
		g.Line3(V3{x, 0, -float32(half) * step}, V3{x, 0, float32(half) * step}, col, cam)
		g.Line3(V3{-float32(half) * step, 0, x}, V3{float32(half) * step, 0, x}, col, cam)
	}
}

func (g *Gfx) Line3(a, b V3, col uint32, cam *Camera) {
	ax, ay, _, ok0 := cam.project(a, g.W, g.H)
	if !ok0 {
		return
	}
	bx, by, _, ok1 := cam.project(b, g.W, g.H)
	if !ok1 {
		return
	}
	g.Line2D(ax, ay, bx, by, col, 1)
}
