//go:build windows

package myc

import (
	"syscall"
	"time"
	"unsafe"
)

var (
	user32   = syscall.NewLazyDLL("user32.dll")
	gdi32    = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	pRegisterClassEx = user32.NewProc("RegisterClassExW")
	pCreateWindowEx  = user32.NewProc("CreateWindowExW")
	pDefWindowProc   = user32.NewProc("DefWindowProcW")
	pShowWindow      = user32.NewProc("ShowWindow")
	pPeekMessage     = user32.NewProc("PeekMessageW")
	pTranslateMsg    = user32.NewProc("TranslateMessage")
	pDispatchMsg     = user32.NewProc("DispatchMessageW")
	pPostQuitMessage = user32.NewProc("PostQuitMessage")
	pDestroyWindow   = user32.NewProc("DestroyWindow")
	pGetClientRect   = user32.NewProc("GetClientRect")
	pInvalidateRect  = user32.NewProc("InvalidateRect")
	pSetWindowText   = user32.NewProc("SetWindowTextW")
	pLoadCursor      = user32.NewProc("LoadCursorW")
	pBeginPaint      = user32.NewProc("BeginPaint")
	pEndPaint        = user32.NewProc("EndPaint")
	pGetDC           = user32.NewProc("GetDC")
	pReleaseDC       = user32.NewProc("ReleaseDC")

	pCreateCompatibleDC = gdi32.NewProc("CreateCompatibleDC")
	pCreateDIBSection   = gdi32.NewProc("CreateDIBSection")
	pSelectObject       = gdi32.NewProc("SelectObject")
	pDeleteObject       = gdi32.NewProc("DeleteObject")
	pDeleteDC           = gdi32.NewProc("DeleteDC")
	pBitBlt             = gdi32.NewProc("BitBlt")
	pSetTextColor       = gdi32.NewProc("SetTextColor")
	pSetBkMode          = gdi32.NewProc("SetBkMode")
	pCreateFont         = gdi32.NewProc("CreateFontA")
	pTextOut            = gdi32.NewProc("TextOutA")

	pGetModuleHandle = kernel32.NewProc("GetModuleHandleW")
	pQueryPerfCounter = kernel32.NewProc("QueryPerformanceCounter")
	pQueryPerfFreq    = kernel32.NewProc("QueryPerformanceFrequency")
)

const (
	wsOverlappedWindow = 0x00CF0000
	wsVisible          = 0x10000000
	cwUseDefault       = 0x80000000
	swShow             = 5
	pmRemove           = 0x0001
	wmDestroy          = 0x0002
	wmClose            = 0x0010
	wmPaint            = 0x000F
	wmKeyDown          = 0x0100
	wmEraseBkgnd       = 0x0014
	vkEscape           = 0x1B
	csHRedraw          = 0x0002
	csVRedraw          = 0x0001
	idcArrow           = 32512
	dibRGBColors       = 0
	srcCopy            = 0x00CC0020
	transparent        = 1
	defaultCharset     = 1
	fwNormal           = 400
)

type wndClassExW struct {
	cbSize        uint32
	style         uint32
	lpfnWndProc   uintptr
	cbClsExtra    int32
	cbWndExtra    int32
	hInstance     uintptr
	hIcon         uintptr
	hCursor       uintptr
	hbrBackground uintptr
	lpszMenuName  *uint16
	lpszClassName *uint16
	hIconSm       uintptr
}

type rect struct{ left, top, right, bottom int32 }

type point struct{ x, y int32 }

type msgT struct {
	hwnd     uintptr
	message  uint32
	wParam   uintptr
	lParam   uintptr
	time     uint32
	pt       point
	lPrivate uint32
}

type paintStruct struct {
	hdc         uintptr
	fErase      int32
	rcPaint     rect
	fRestore    int32
	fIncUpdate  int32
	rgbReserved [32]byte
}

type bitmapInfoHeader struct {
	biSize          uint32
	biWidth         int32
	biHeight        int32
	biPlanes        uint16
	biBitCount      uint16
	biCompression   uint32
	biSizeImage     uint32
	biXPelsPerMeter int32
	biYPelsPerMeter int32
	biClrUsed       uint32
	biClrImportant  uint32
}

type bitmapInfo struct {
	bmiHeader bitmapInfoHeader
	bmiColors [1]uint32
}

// App describes a native MYCEL window.
type App struct {
	Title  string
	Width  int
	Height int
	Frame  func(g *Gfx, t float64, dt float64)
}

var (
	app      *App
	gfx      *Gfx
	memDC    uintptr
	hBitmap  uintptr
	fonts    = map[int]uintptr{}
	texts    []textItem
	lastTick int64
	freq     int64
)

func utf16p(s string) *uint16 {
	p, err := syscall.UTF16PtrFromString(s)
	if err != nil {
		return nil
	}
	return p
}

func wndProc(hwnd, msg, wparam, lparam uintptr) uintptr {
	switch msg {
	case wmClose:
		pDestroyWindow.Call(hwnd)
		return 0
	case wmDestroy:
		pPostQuitMessage.Call(0)
		return 0
	case wmEraseBkgnd:
		return 1
	case wmKeyDown:
		if wparam == vkEscape {
			pDestroyWindow.Call(hwnd)
		}
		return 0
	}
	r, _, _ := pDefWindowProc.Call(hwnd, msg, wparam, lparam)
	return r
}

func makeFont(size int) uintptr {
	if h, ok := fonts[size]; ok {
		return h
	}
	name := utf16p("Consolas")
	h, _, _ := pCreateFont.Call(
		uintptr(-int32(size)), 0, 0, 0, fwNormal, 0, 0, 0,
		defaultCharset, 0, 0, 4, 0, uintptr(unsafe.Pointer(name)),
	)
	fonts[size] = h
	return h
}

func (g *Gfx) Text(x, y int, s string, col uint32, size int) {
	texts = append(texts, textItem{x, y, s, col, size})
}

func flushText() {
	if len(texts) == 0 {
		return
	}
	pSetBkMode.Call(memDC, transparent)
	for _, t := range texts {
		font := makeFont(t.size)
		old, _, _ := pSelectObject.Call(memDC, font)
		// COLORREF is 0x00BBGGRR
		cr := (t.c&0xFF)<<16 | (t.c & 0xFF00) | ((t.c >> 16) & 0xFF)
		pSetTextColor.Call(memDC, uintptr(cr))
		b := []byte(t.s)
		if len(b) > 0 {
			pTextOut.Call(memDC, uintptr(t.x), uintptr(t.y),
				uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)))
		}
		pSelectObject.Call(memDC, old)
	}
	texts = texts[:0]
}

// Run opens the window and drives the frame callback at ~60 Hz.
func Run(a *App) {
	app = a
	hInst, _, _ := pGetModuleHandle.Call(0)
	className := utf16p("MYCELNativeWindow")
	cursor, _, _ := pLoadCursor.Call(0, idcArrow)

	wc := wndClassExW{
		style:         csHRedraw | csVRedraw,
		lpfnWndProc:   syscall.NewCallback(wndProc),
		hInstance:     hInst,
		hCursor:       cursor,
		lpszClassName: className,
	}
	wc.cbSize = uint32(unsafe.Sizeof(wc))
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	r := rect{0, 0, int32(a.Width), int32(a.Height)}
	style := uintptr(wsOverlappedWindow)
	user32.NewProc("AdjustWindowRect").Call(uintptr(unsafe.Pointer(&r)), style, 0)

	hwnd, _, _ := pCreateWindowEx.Call(
		0, uintptr(unsafe.Pointer(className)), uintptr(unsafe.Pointer(utf16p(a.Title))),
		style|wsVisible,
		cwUseDefault, cwUseDefault,
		uintptr(r.right-r.left), uintptr(r.bottom-r.top),
		0, 0, hInst, 0,
	)
	if hwnd == 0 {
		panic("mycel: cannot create window")
	}
	pShowWindow.Call(hwnd, swShow)

	screenDC, _, _ := pGetDC.Call(hwnd)
	memDC, _, _ = pCreateCompatibleDC.Call(screenDC)

	bmi := bitmapInfo{}
	bmi.bmiHeader.biSize = uint32(unsafe.Sizeof(bmi.bmiHeader))
	bmi.bmiHeader.biWidth = int32(a.Width)
	bmi.bmiHeader.biHeight = -int32(a.Height) // top-down
	bmi.bmiHeader.biPlanes = 1
	bmi.bmiHeader.biBitCount = 32
	bmi.bmiHeader.biCompression = dibRGBColors

	var bits unsafe.Pointer
	hBitmap, _, _ = pCreateDIBSection.Call(memDC, uintptr(unsafe.Pointer(&bmi)),
		dibRGBColors, uintptr(unsafe.Pointer(&bits)), 0, 0)
	pSelectObject.Call(memDC, hBitmap)

	gfx = &Gfx{
		W:   a.Width,
		H:   a.Height,
		Pix: unsafe.Slice((*uint32)(bits), a.Width*a.Height),
		Z:   make([]float32, a.Width*a.Height),
	}

	pQueryPerfFreq.Call(uintptr(unsafe.Pointer(&freq)))
	pQueryPerfCounter.Call(uintptr(unsafe.Pointer(&lastTick)))

	var m msgT
	t0 := time.Now()
	const target = time.Second / 60

	for {
		for {
			got, _, _ := pPeekMessage.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0, pmRemove)
			if got == 0 {
				break
			}
			if m.message == 0x0012 { // WM_QUIT
				return
			}
			pTranslateMsg.Call(uintptr(unsafe.Pointer(&m)))
			pDispatchMsg.Call(uintptr(unsafe.Pointer(&m)))
		}

		var now int64
		pQueryPerfCounter.Call(uintptr(unsafe.Pointer(&now)))
		dt := float64(now-lastTick) / float64(freq)
		lastTick = now
		if dt > 0.1 {
			dt = 0.1
		}

		gfx.Clear(RGB(7, 12, 18))
		a.Frame(gfx, time.Since(t0).Seconds(), dt)
		flushText()

		wdc, _, _ := pGetDC.Call(hwnd)
		pBitBlt.Call(wdc, 0, 0,
			uintptr(a.Width), uintptr(a.Height), memDC, 0, 0, srcCopy)
		pReleaseDC.Call(hwnd, wdc)

		if elapsed := time.Since(t0); false {
			_ = elapsed
		}
		time.Sleep(target)
	}
}
