package myc

// Scene ties a body, a camera and a render style together. The generated
// native program is little more than a Scene plus a window.
type Scene struct {
	Body     *Biped
	Cam      *Camera
	Grid     bool
	TrailLen int
	FollowX  float32
	LookY    float32
	Panel    bool
	PanelW   int

	time  float64
	trail [][6][2]float64
	brain BrainView
}

// Step advances the world by dt and draws one frame.
func (s *Scene) Step(g *Gfx, dt float64) {
	if s.Body == nil {
		return
	}
	s.Body.Update(dt)
	s.time += dt

	if s.Cam != nil {
		x := float32(s.Body.X())
		if s.FollowX != 0 {
			s.Cam.Pos.X = x + s.FollowX
			s.Cam.Target.X = x
		}
		if s.LookY != 0 {
			s.Cam.Target.Y = s.LookY
		}
		if s.Panel {
			s.Cam.Viewport(0, 0, g.W-s.PanelW, g.H)
		} else {
			s.Cam.Update(g.W, g.H)
		}
	}

	if s.TrailLen > 0 {
		s.trail = append(s.trail, s.Body.Positions())
		if len(s.trail) > s.TrailLen {
			s.trail = s.trail[1:]
		}
	}

	if s.Grid {
		g.Grid(s.Cam, 14, 1.0, ColGrid)
	}
	for _, foot := range []int{FootL, FootR} {
		if s.Body.contact[(foot-FootL)/2] {
			c := V3{float32(s.Body.pos[foot][0]), 0.002, 0}
			g.Disc3(c, 0.16, 0.004, RGB(24, 54, 68), s.Cam)
		}
	}
	for _, ghost := range s.trail {
		ghostBeams(g, s.Cam, ghost, 0.3)
	}
	beams(g, s.Cam, s.Body.pos, 1.0)
	for i, col := range []uint32{ColJoint, ColJoint, ColJoint, ColFoot, ColJoint, ColFoot} {
		p := V3{float32(s.Body.pos[i][0]), float32(s.Body.pos[i][1]), 0}
		sz := float32(0.055)
		if i == Hip {
			sz = 0.07
		}
		g.Box(p, sz, sz, 0.20, col, s.Cam)
	}
	head := V3{float32(s.Body.pos[Shoulder][0]), float32(s.Body.pos[Shoulder][1] + 0.115), 0}
	g.Box(head, 0.17, 0.17, 0.17, ColTorso, s.Cam)

	if s.Panel {
		s.brain.Draw(g, s.Body.Ctrl(), g.W-s.PanelW, 0, s.PanelW, g.H)
		HudNarrow(g, s.Body, s.time, g.W-s.PanelW)
	} else {
		Hud(g, s.Body, s.time, "software raster  -  ESC to quit")
	}
}
