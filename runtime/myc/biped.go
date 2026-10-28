package myc

import "math"

// Joint indices.
const (
	Shoulder = 0
	Hip      = 1
	KneeL    = 2
	FootL    = 3
	KneeR    = 4
	FootR    = 5
)

var bones = [7][2]int{
	{Shoulder, Hip}, {Hip, KneeL}, {KneeL, FootL},
	{Hip, KneeR}, {KneeR, FootR},
	{Hip, FootL}, {Hip, FootR},
}

var boneLen = [7]float64{0.50, 0.42, 0.40, 0.42, 0.40, 0.760, 0.760}
var invMass = [6]float64{1.0 / 6.0, 1.0 / 6.0, 1.0 / 1.4, 1.0, 1.0 / 1.4, 1.0}

const (
	kneeStance  = 0.90
	kneeSwing   = 0.02
	gravity     = -9.81
	liftHigh    = 0.055
	liftLow     = 0.015
	minPhase    = 0.075
	substeps    = 3
	iterations  = 5
	torsoGain   = 0.25
	thrustGain  = 6.0
	maxSpeed    = 8.0
	damping     = 0.9995
	cells       = 24
	inputs      = 14
	outputs     = 8
	phaseFeats  = 4
	tauNeuron   = 0.040
	cpgHz       = 1.60
	paramStride = 0
)

// Controller is the compiled liquid tissue: 24 cells, 14 inputs, 8 outputs.
type Controller struct {
	w      [cells][cells]float64
	u      [cells][inputs]float64
	b      [cells]float64
	wout   [outputs][cells]float64
	gph    [outputs][phaseFeats]float64
	bout   [outputs]float64
	h0     [cells]float64
	h       [cells]float64
	lastIn  [inputs]float64
	lastOut [outputs]float64
	phi     float64
	theta  []float32
	inited bool
}

func (c *Controller) load(theta []float32) {
	if c.inited {
		return
	}
	c.theta = theta
	i := 0
	next := func() float64 { v := float64(theta[i]); i++; return v }
	for a := 0; a < cells; a++ {
		for b := 0; b < cells; b++ {
			c.w[a][b] = next()
		}
	}
	for a := 0; a < cells; a++ {
		for b := 0; b < inputs; b++ {
			c.u[a][b] = next()
		}
	}
	for a := 0; a < cells; a++ {
		c.b[a] = next()
	}
	for a := 0; a < outputs; a++ {
		for b := 0; b < cells; b++ {
			c.wout[a][b] = next()
		}
	}
	for a := 0; a < outputs; a++ {
		for b := 0; b < phaseFeats; b++ {
			c.gph[a][b] = next()
		}
	}
	for a := 0; a < outputs; a++ {
		c.bout[a] = next()
	}
	c.inited = true
}

func (c *Controller) reset() {
	c.h = c.h0
	c.phi = 0
}

func (c *Controller) deriv(h *[cells]float64, x *[inputs]float64, out *[cells]float64) {
	var z [cells]float64
	for a := 0; a < cells; a++ {
		s := c.b[a]
		for b := 0; b < cells; b++ {
			s += c.w[a][b] * h[b]
		}
		for b := 0; b < inputs; b++ {
			s += c.u[a][b] * x[b]
		}
		z[a] = math.Tanh(s)
	}
	for a := 0; a < cells; a++ {
		out[a] = (-h[a] + z[a]) / tauNeuron
	}
}

// forward integrates the tissue one tick and returns the command vector.
func (c *Controller) forward(dt float64, x *[inputs]float64) [outputs]float64 {
	c.lastIn = *x
	var k1, k2, k3, k4, tmp [cells]float64
	c.deriv(&c.h, x, &k1)
	for a := 0; a < cells; a++ {
		tmp[a] = c.h[a] + 0.5*dt*k1[a]
	}
	c.deriv(&tmp, x, &k2)
	for a := 0; a < cells; a++ {
		tmp[a] = c.h[a] + 0.5*dt*k2[a]
	}
	c.deriv(&tmp, x, &k3)
	for a := 0; a < cells; a++ {
		tmp[a] = c.h[a] + dt*k3[a]
	}
	c.deriv(&tmp, x, &k4)
	for a := 0; a < cells; a++ {
		c.h[a] += dt / 6 * (k1[a] + 2*k2[a] + 2*k3[a] + k4[a])
	}

	c.phi += 2 * math.Pi * cpgHz * dt
	s, co := math.Sin(c.phi), math.Cos(c.phi)
	relu := func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}
	ph := [phaseFeats]float64{s, co, relu(s), relu(-s)}

	var y [outputs]float64
	for a := 0; a < outputs; a++ {
		v := c.bout[a]
		for b := 0; b < cells; b++ {
			v += c.wout[a][b] * c.h[b]
		}
		for b := 0; b < phaseFeats; b++ {
			v += c.gph[a][b] * ph[b]
		}
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		y[a] = v
	}
	c.lastOut = y
	return y
}

// ---------------------------------------------------------------------------
// body
// ---------------------------------------------------------------------------

type Biped struct {
	pos, prev, vel [6][2]float64
	pin            [2][2]float64
	swing          [2]bool
	phaseTimer     [2]float64
	cmd            [8]float64
	targets        [4]float64
	contact        [2]bool
	fallen         bool
	time           float64
	dt             float64
	ctl            Controller
	prevAngles     [4]float64
	anglesReady    bool
}

func NewBiped(theta []float32) *Biped {
	b := &Biped{dt: 1.0 / 240.0}
	b.ctl.load(theta)
	b.Reset()
	return b
}

func (b *Biped) Reset() {
	b.pos = [6][2]float64{}
	b.pos[Shoulder] = [2]float64{0, 0.700 + 0.50}
	b.pos[Hip] = [2]float64{0, 0.700}
	b.pos[KneeL] = kneeOf(b.pos[Hip], [2]float64{-0.165, 0})
	b.pos[FootL] = [2]float64{-0.165, 0}
	b.pos[KneeR] = kneeOf(b.pos[Hip], [2]float64{0.165, 0})
	b.pos[FootR] = [2]float64{0.165, 0}
	b.prev = b.pos
	b.vel = [6][2]float64{}
	b.pin[0] = b.pos[FootL]
	b.pin[1] = b.pos[FootR]
	b.swing = [2]bool{false, false}
	b.phaseTimer = [2]float64{}
	b.fallen = false
	b.time = 0
	b.anglesReady = false
	b.ctl.reset()
}

func kneeOf(hip, foot [2]float64) [2]float64 {
	const l1, l2 = 0.42, 0.40
	d := [2]float64{foot[0] - hip[0], foot[1] - hip[1]}
	l := math.Hypot(d[0], d[1])
	if l < 1e-6 {
		l = 1e-6
	}
	if l > l1+l2-1e-4 {
		l = l1 + l2 - 1e-4
	}
	u := [2]float64{d[0] / l, d[1] / l}
	a := (l1*l1 - l2*l2 + l*l) / (2 * l)
	h := math.Sqrt(math.Max(l1*l1-a*a, 0))
	perp := [2]float64{u[1], -u[0]}
	if perp[0] < 0 {
		perp[0], perp[1] = -perp[0], -perp[1]
	}
	return [2]float64{hip[0] + a*u[0] + h*perp[0], hip[1] + a*u[1] + h*perp[1]}
}

func (b *Biped) angle(a, c [2]float64) float64 {
	d := [2]float64{c[0] - a[0], c[1] - a[1]}
	return math.Atan2(d[0], -d[1])
}

func rot(pos *[6][2]float64, pivot int, members []int, phi float64) {
	s, c := math.Sin(phi), math.Cos(phi)
	px, py := pos[pivot][0], pos[pivot][1]
	for _, m := range members {
		dx, dy := pos[m][0]-px, pos[m][1]-py
		pos[m][0] = px + dx*c - dy*s
		pos[m][1] = py + dx*s + dy*c
	}
}

func (b *Biped) Step() {
	h := b.dt / substeps

	lift := [2]float64{b.cmd[1], b.cmd[3]}

	for j := 0; j < 2; j++ {
		want := lift[j] > liftHigh
		if b.swing[j] {
			want = lift[j] > liftLow
		}
		if want != b.swing[j] && b.phaseTimer[j] < minPhase {
			want = b.swing[j]
		}
		if want != b.swing[j] {
			b.phaseTimer[j] = 0
		} else {
			b.phaseTimer[j] += b.dt
		}
		b.swing[j] = want
	}
	if b.swing[0] && b.swing[1] {
		if lift[0] < lift[1] {
			b.swing[0] = false
		} else {
			b.swing[1] = false
		}
	}
	for j, foot := range []int{FootL, FootR} {
		if !b.swing[j] {
			b.pin[j] = b.pos[foot]
		}
	}

	feet := [2]int{FootL, FootR}
	for s := 0; s < substeps; s++ {
		for i := 0; i < 6; i++ {
			b.vel[i][1] += gravity * h
		}
		thrust := 0.0
		if !b.swing[0] {
			thrust += b.cmd[5]
		}
		if !b.swing[1] {
			thrust += b.cmd[6]
		}
		b.vel[Hip][0] += thrustGain * thrust * h

		b.prev = b.pos
		for i := 0; i < 6; i++ {
			b.pos[i][0] += b.vel[i][0] * h
			b.pos[i][1] += b.vel[i][1] * h
		}

		for it := 0; it < iterations; it++ {
			for bi := 0; bi < 7; bi++ {
				stiff := 1.0
				if bi >= 5 {
					j := bi - 5
					if b.swing[j] {
						stiff = kneeSwing
					} else {
						stiff = kneeStance
					}
				}
				a := bones[bi][0]
				c := bones[bi][1]
				dx := b.pos[c][0] - b.pos[a][0]
				dy := b.pos[c][1] - b.pos[a][1]
				dist := math.Hypot(dx, dy) + 1e-9
				err := (dist - boneLen[bi]) * stiff
				w := invMass[a] + invMass[c]
				cx := dx / dist * err / w
				cy := dy / dist * err / w
				b.pos[a][0] += invMass[a] * cx
				b.pos[a][1] += invMass[a] * cy
				b.pos[c][0] -= invMass[c] * cx
				b.pos[c][1] -= invMass[c] * cy
			}

			limit := 2.6 * h
			for j := 0; j < 2; j++ {
				foot := feet[j]
				var tx, ty float64
				if b.swing[j] {
					tx = b.pos[Hip][0] + b.cmd[2*j]
					ty = b.cmd[2*j+1]
					dx := clamp(tx-b.pos[foot][0], -limit, limit)
					dy := clamp(ty-b.pos[foot][1], -limit, limit)
					b.pos[foot][0] += dx
					b.pos[foot][1] += dy
				} else {
					b.pos[foot][0] = b.pin[j][0]
					b.pos[foot][1] = 0
				}
			}

			for _, kn := range [2][2]int{{Hip, KneeL}, {Hip, KneeR}} {
				knee := kn[1]
				foot := knee + 1
				thigh := b.angle(b.pos[kn[0]], b.pos[knee])
				shank := b.angle(b.pos[knee], b.pos[foot])
				bend := thigh - shank
				if bend < 0 {
					rot(&b.pos, knee, []int{foot}, -0.5*bend)
				}
			}

			upx := b.pos[Shoulder][0] - b.pos[Hip][0]
			upy := b.pos[Shoulder][1] - b.pos[Hip][1]
			theta := math.Atan2(upx, upy)
			rot(&b.pos, Hip, []int{Shoulder}, torsoGain*clamp(theta-b.cmd[4], -0.12, 0.12))

			for i := 0; i < 6; i++ {
				if b.pos[i][1] < 0 {
					b.pos[i][1] = 0
				}
			}
		}

		touch := [6]bool{}
		for i := 0; i < 6; i++ {
			touch[i] = b.pos[i][1] <= 1e-4
		}
		for i := 0; i < 6; i++ {
			if touch[i] {
				keep := 0.0
				if i == KneeL || i == KneeR {
					keep = 0.15
				}
				dx := b.pos[i][0] - b.prev[i][0]
				b.pos[i][0] = b.prev[i][0] + dx*keep
			}
		}

		for i := 0; i < 6; i++ {
			b.vel[i][0] = (b.pos[i][0] - b.prev[i][0]) / h
			b.vel[i][1] = (b.pos[i][1] - b.prev[i][1]) / h
			b.vel[i][0] = clamp(b.vel[i][0], -maxSpeed, maxSpeed)
			b.vel[i][1] = clamp(b.vel[i][1], -maxSpeed, maxSpeed)
			b.vel[i][0] *= damping
			b.vel[i][1] *= damping
		}
	}

	b.time += b.dt
	b.contact[0] = b.pos[FootL][1] < 0.02
	b.contact[1] = b.pos[FootR][1] < 0.02
	if b.time > 0.40 {
		pitch := math.Abs(b.TorsoPitch())
		if b.pos[Hip][1] < 0.50 || pitch > 1.2 {
			b.fallen = true
		}
	}
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func (b *Biped) TorsoPitch() float64 {
	upx := b.pos[Shoulder][0] - b.pos[Hip][0]
	upy := b.pos[Shoulder][1] - b.pos[Hip][1]
	return math.Atan2(upx, upy)
}

// Update runs one frame worth of simulation and returns the current command.
func (b *Biped) Update(dt float64) {
	steps := int(math.Round(dt / b.dt))
	if steps < 1 {
		steps = 1
	}
	if steps > 8 {
		steps = 8
	}
	for i := 0; i < steps; i++ {
		b.tick()
	}
}

func (b *Biped) tick() {
	var x [inputs]float64
	pitch := b.TorsoPitch()
	upx := (b.pos[Shoulder][0] - b.pos[Hip][0]) - (b.prev[Shoulder][0] - b.prev[Hip][0])
	upy := (b.pos[Shoulder][1] - b.pos[Hip][1]) - (b.prev[Shoulder][1] - b.prev[Hip][1])
	dpitch := math.Atan2(upx, upy) / b.dt

	tl := b.angle(b.pos[Hip], b.pos[KneeL])
	sl := b.angle(b.pos[KneeL], b.pos[FootL])
	tr := b.angle(b.pos[Hip], b.pos[KneeR])
	sr := b.angle(b.pos[KneeR], b.pos[FootR])
	ang := [4]float64{tl, sl - tl, tr, sr - tr}

	var rate [4]float64
	if b.anglesReady {
		for i := 0; i < 4; i++ {
			rate[i] = (ang[i] - b.prevAngles[i]) / b.dt
		}
	}
	b.prevAngles = ang
	b.anglesReady = true

	cL, cR := 0.0, 0.0
	if b.contact[0] {
		cL = 1
	}
	if b.contact[1] {
		cR = 1
	}
	s, co := math.Sin(b.ctl.phi), math.Cos(b.ctl.phi)

	x[0] = pitch
	x[1] = dpitch
	x[2] = math.Sin(ang[0])
	x[3] = math.Sin(ang[1])
	x[4] = math.Sin(ang[2])
	x[5] = math.Sin(ang[3])
	x[6] = rate[0]
	x[7] = rate[1]
	x[8] = rate[2]
	x[9] = rate[3]
	x[10] = cL
	x[11] = cR
	x[12] = s
	x[13] = co

	y := b.ctl.forward(b.dt, &x)
	b.cmd = y
	b.targets[0] = 0.30 * y[0]
	b.targets[1] = 0.12 * math.Max(y[1]+0.12, 0) / 1.12
	b.targets[2] = 0.30 * y[2]
	b.targets[3] = 0.12 * math.Max(y[3]+0.12, 0) / 1.12
	b.cmd[0] = b.targets[0]
	b.cmd[1] = b.targets[1]
	b.cmd[2] = b.targets[2]
	b.cmd[3] = b.targets[3]
	b.cmd[4] = 0.28 * y[4]
	b.cmd[5] = y[5]
	b.cmd[6] = y[6]

	b.Step()
}

func (b *Biped) Pos(i int) [2]float64 { return b.pos[i] }
func (b *Biped) Positions() [6][2]float64 { return b.pos }
func (b *Biped) Ctrl() *Controller      { return &b.ctl }
func (b *Biped) X() float64           { return b.pos[Hip][0] }
func (b *Biped) SpeedX() float64      { return b.vel[Hip][0] }
func (b *Biped) Fallen() bool         { return b.fallen }
func (b *Biped) Command() [8]float64  { return b.cmd }
func (b *Biped) Swing() [2]bool       { return b.swing }
