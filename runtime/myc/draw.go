package myc

import (
	"fmt"
	"math"
)

var (
	ColBg      = RGB(7, 12, 18)
	ColGround  = RGB(20, 44, 56)
	ColGrid    = RGB(16, 34, 44)
	ColTorso   = RGB(183, 140, 255)
	ColLimb    = RGB(79, 214, 233)
	ColJoint   = RGB(111, 233, 212)
	ColFoot    = RGB(255, 184, 107)
	ColHud     = RGB(201, 222, 232)
	ColHudDim  = RGB(94, 122, 138)
	ColAccent  = RGB(111, 233, 212)
)

// DrawBiped renders the body in 3D, extruded along Z.
func DrawBiped(g *Gfx, cam *Camera, b *Biped, trail [][6][2]float64) {
	g.Grid(cam, 14, 1.0, ColGrid)

	// contact shadow
	for i, foot := range []int{FootL, FootR} {
		if b.contact[i] {
			c := V3{float32(b.pos[foot][0]), 0.002, 0}
			g.Disc3(c, 0.16, 0.005, RGB(26, 58, 72), cam)
		}
	}

	for _, ghost := range trail {
		ghostBeams(g, cam, ghost, 0.06)
	}

	beams(g, cam, b.pos, 1.0)

	// joints
	for i, col := range []uint32{ColJoint, ColJoint, ColJoint, ColFoot, ColJoint, ColFoot} {
		p := V3{float32(b.pos[i][0]), float32(b.pos[i][1]), 0}
		s := float32(0.055)
		if i == Hip {
			s = 0.07
		}
		g.Box(p, s, s, 0.20, col, cam)
	}
	head := V3{float32(b.pos[Shoulder][0]), float32(b.pos[Shoulder][1] + 0.115), 0}
	g.Box(head, 0.17, 0.17, 0.17, ColTorso, cam)
}

func beams(g *Gfx, cam *Camera, p [6][2]float64, alpha float32) {
	beam := func(a, b int, w float64, col uint32) {
		g.Beam(
			V3{float32(p[a][0]), float32(p[a][1]), 0},
			V3{float32(p[b][0]), float32(p[b][1]), 0},
			float32(w), 0.22, col, cam,
		)
	}
	beam(Shoulder, Hip, 0.17, ColTorso)
	beam(Hip, KneeL, 0.115, ColLimb)
	beam(KneeL, FootL, 0.095, ColLimb)
	beam(Hip, KneeR, 0.115, ColLimb)
	beam(KneeR, FootR, 0.095, ColLimb)
}

func ghostBeams(g *Gfx, cam *Camera, p [6][2]float64, k float32) {
	col := shade(ColLimb, 0.28)
	torso := shade(ColTorso, 0.28)
	g.Beam(V3{float32(p[Shoulder][0]), float32(p[Shoulder][1]), 0},
		V3{float32(p[Hip][0]), float32(p[Hip][1]), 0}, 0.17, 0.22, torso, cam)
	for _, s := range [][2]int{{Hip, KneeL}, {KneeL, FootL}, {Hip, KneeR}, {KneeR, FootR}} {
		g.Beam(V3{float32(p[s[0]][0]), float32(p[s[0]][1]), 0},
			V3{float32(p[s[1]][0]), float32(p[s[1]][1]), 0}, 0.10, 0.22, col, cam)
	}
}

// Disc3 draws a flat disc on the ground plane.
func (g *Gfx) Disc3(c V3, r float32, y float32, col uint32, cam *Camera) {
	const n = 14
	for i := 0; i < n; i++ {
		a0 := float32(i) / n * 2 * math.Pi
		a1 := float32(i+1) / n * 2 * math.Pi
		p0 := V3{c.X + r*float32(math.Cos(float64(a0))), y, c.Z + r*float32(math.Sin(float64(a0)))}
		p1 := V3{c.X + r*float32(math.Cos(float64(a1))), y, c.Z + r*float32(math.Sin(float64(a1)))}
		g.Tri(c, p0, p1, col, cam)
	}
}

// Hud draws the heads-up display.
func Hud(g *Gfx, b *Biped, t float64, extra string) {
	g.Text(24, 18, "MYCEL  /  native render", ColHud, 20)
	g.Text(24, 44, "no Python  -  single binary  -  software raster + z-buffer", ColHudDim, 13)

	g.Text(g.W-210, 18, fmt.Sprintf("%6.2f m", b.X()), ColAccent, 28)
	g.Text(g.W-210, 58, fmt.Sprintf("%5.2f m/s", b.SpeedX()), ColLimb, 16)
	g.Text(g.W-210, 82, fmt.Sprintf("t = %5.2f s", t), ColHudDim, 13)
	if extra != "" {
		g.Text(24, g.H-34, extra, ColHudDim, 13)
	}
}

// HudNarrow draws the HUD inside a viewport of the given width.
func HudNarrow(g *Gfx, b *Biped, t float64, width int) {
	g.Text(24, 18, "MYCEL  /  native render", ColHud, 20)
	g.Text(24, 44, "no Python  -  single binary  -  software raster + z-buffer", ColHudDim, 13)
	g.Text(width-210, 18, fmt.Sprintf("%6.2f m", b.X()), ColAccent, 28)
	g.Text(width-210, 58, fmt.Sprintf("%5.2f m/s", b.SpeedX()), ColLimb, 16)
	g.Text(width-210, 82, fmt.Sprintf("t = %5.2f s", t), ColHudDim, 13)
	g.Text(24, g.H-30, "ESC to quit  -  the right panel is the live tissue", ColHudDim, 12)
}
