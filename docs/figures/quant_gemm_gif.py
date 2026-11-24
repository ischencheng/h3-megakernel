"""Quantize -> GEMM dataflow animation. Usage: python quant_gemm_gif.py OUT.gif [--frames DIR t1 t2 ...]"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "C:/Windows/Fonts/seguisb.ttf"
FONTS = {}
LABEL_C = (100, 116, 139)

W, H, SS, FPS = 1256, 760, 2, 25          # 15.7 x 9.5 cm at 80 px/cm
T_END = 10.4


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back_out(x):
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def hexc(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BG = WHITE
UNIT_C = [hexc("#2563EB"), hexc("#0EA5E9"), hexc("#7C3AED"), hexc("#C026D3")]
RAW_LO, RAW_HI = hexc("#E2E8F0"), hexc("#475569")
GHOST = hexc("#E2E8F0")
GMEM_FILL, GMEM_EDGE = hexc("#F1F5F9"), hexc("#94A3B8")
CTR_PENDING_FILL, CTR_PENDING_EDGE = WHITE, hexc("#94A3B8")
CTR_DONE_FILL, CTR_DONE_EDGE = hexc("#22C55E"), hexc("#15803D")
ARROW_C = hexc("#EF4444")
SHADOW_C = hexc("#0F172A")

# style = (fill, border, glow colour, glow alpha)
IDLE = (hexc("#F8FAFC"), hexc("#CBD5E1"), hexc("#CBD5E1"), 0.0)
GREEN = (hexc("#DCFCE7"), hexc("#22C55E"), hexc("#22C55E"), 0.55)
YELLOW = (hexc("#FEF9C3"), hexc("#EAB308"), hexc("#FACC15"), 0.40)
RED = (hexc("#FEE2E2"), hexc("#EF4444"), hexc("#EF4444"), 0.50)
GREY = (hexc("#E5E7EB"), hexc("#9CA3AF"), hexc("#9CA3AF"), 0.0)


def blend_style(a, b, t):
    return (mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t), lerp(a[3], b[3], t))


# ---------------------------------------------------------------- layout (world units = output px at full zoom-out)
SM, GAP_X, GAP_Y = 170, 36, 44
GX0, TOP_Y = 420, 104
GMEM = (GX0, 540, GX0 + 4 * SM + 3 * GAP_X, 656)
SLOT_CY = 598
DATA_W, DATA_H, CTR = 112, 38, 38
SLOT_GAP = 10


def sm_box(row, col):
    x = GX0 + col * (SM + GAP_X)
    y = TOP_Y + row * (SM + GAP_Y)
    return (x, y, x + SM, y + SM)


def sm_center(row, col):
    b = sm_box(row, col)
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


PAIR_W = DATA_W + SLOT_GAP + CTR
SLOT_PAD = ((GMEM[2] - GMEM[0]) - 4 * PAIR_W) / 5


def slot_x0(u):
    return GMEM[0] + SLOT_PAD + u * (PAIR_W + SLOT_PAD)


def data_center(u):
    return (slot_x0(u) + DATA_W / 2, SLOT_CY)


def ctr_center(u):
    return (slot_x0(u) + DATA_W + SLOT_GAP + CTR / 2, SLOT_CY)


ROWS, COLS = 8, 32
UW, UH = 150.0, 37.5                       # one work unit: 4 x 16 cells
MAT_X0, MAT_Y0 = 48.0, 380 - 2 * UH
SPLIT_GAP = 12.0


def unit_home(u, split):
    r, k = divmod(u, 2)
    return (MAT_X0 + UW / 2 + k * UW + (k - 0.5) * SPLIT_GAP * split,
            MAT_Y0 + UH / 2 + r * UH + (r - 0.5) * SPLIT_GAP * split)


# ---------------------------------------------------------------- data
rng = np.random.RandomState(11)
yy, xx = np.mgrid[0:ROWS, 0:COLS]
VALS = np.clip(0.55 * rng.rand(ROWS, COLS) + 0.45 * (0.5 + 0.5 * np.sin(xx * 0.33 + yy * 0.9)), 0, 1)


def unit_vals(u):
    r, k = divmod(u, 2)
    return VALS[r * 4:(r + 1) * 4, k * 16:(k + 1) * 16]


def tinted(u, v):
    return mix(mix(UNIT_C[u], WHITE, 0.82), mix(UNIT_C[u], BLACK, 0.22), v)


RAW = [[[mix(RAW_LO, RAW_HI, v) for v in row] for row in unit_vals(u)] for u in range(4)]
TINT = [[[tinted(u, v) for v in row] for row in unit_vals(u)] for u in range(4)]
POST = [[[tinted(u, round(v * 3) / 3) for v in row] for row in unit_vals(u)] for u in range(4)]


def mix_grid(g1, g2, t):
    return [[mix(a, b, t) for a, b in zip(r1, r2)] for r1, r2 in zip(g1, g2)]


def shimmer(grid, seed, amp):
    n = np.random.RandomState(seed).rand(4, 16) ** 3
    return [[mix(c, WHITE, amp * n[i, j]) for j, c in enumerate(row)] for i, row in enumerate(grid)]


# ---------------------------------------------------------------- timeline (seconds)
T_SPLIT = (0.6, 1.4)
T_ZOOM = (1.5, 2.8)
T_RED = (2.6, 3.0)
LAUNCH = [3.0 + 0.22 * u for u in range(4)]
FLY = 0.65
ARRIVE = [l + FLY for l in LAUNCH]
CRUNCH = [1.6, 2.2, 0.8, 1.2]
FINISH = [a + c for a, c in zip(ARRIVE, CRUNCH)]          # SM done computing, starts writing
WRITE = 0.75
LANDED = [f + WRITE for f in FINISH]                       # data in gmem, counter green
READY = [max(LANDED[0], LANDED[1]) + 0.15, max(LANDED[2], LANDED[3]) + 0.15]  # per bottom SM
ARROW_T = 0.35
READ_START = [r + ARROW_T + 0.12 for r in READY]
READ_T = 0.7
READ_END = [r + READ_T for r in READ_START]
T_LOOP = (T_END - 0.6, T_END)

Z0 = 3.2
C0 = (MAT_X0 + UW, 380.0)
C1 = (W / 2, H / 2)


def camera(t):
    e = ease(prog(t, *T_ZOOM))
    s = math.exp(lerp(math.log(Z0), 0.0, e))
    w = (1 / s - 1 / Z0) / (1 - 1 / Z0)
    return (lerp(C0[0], C1[0], w), lerp(C0[1], C1[1], w), s)


# ---------------------------------------------------------------- drawing
class Frame:
    def __init__(self, t):
        self.t = t
        self.cx, self.cy, self.s = camera(t)
        self.img = Image.new("RGB", (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def P(self, x, y):
        return ((x - self.cx) * self.s + W / 2) * SS, ((y - self.cy) * self.s + H / 2) * SS

    def L(self, v):
        return v * self.s * SS

    def box(self, b):
        x0, y0 = self.P(b[0], b[1])
        x1, y1 = self.P(b[2], b[3])
        return [x0, y0, x1, y1]

    def rrect(self, b, r, fill=None, edge=None, width=0, alpha=1.0):
        a = int(255 * alpha)
        self.d.rounded_rectangle(self.box(b), radius=self.L(r),
                                 fill=None if fill is None else fill + (a,),
                                 outline=None if edge is None else edge + (a,),
                                 width=max(1, int(round(self.L(width)))) if edge is not None else 0)

    def glow(self, b, r, color, alpha, spread):
        if alpha <= 0.01:
            return
        x0, y0, x1, y1 = self.box(b)
        m = int(self.L(spread) * 2.6) + 2
        f = 4
        bw, bh = int(x1 - x0) + 2 * m, int(y1 - y0) + 2 * m
        mask = Image.new("L", (bw // f + 1, bh // f + 1), 0)
        ImageDraw.Draw(mask).rounded_rectangle([m / f, m / f, (bw - m) / f, (bh - m) / f],
                                               radius=self.L(r) / f, fill=int(255 * clamp(alpha)))
        mask = mask.filter(ImageFilter.GaussianBlur(self.L(spread) / f)).resize((bw, bh), Image.BILINEAR)
        self.img.paste(Image.new("RGB", (bw, bh), color), (int(x0) - m, int(y0) - m), mask)

    def shadow(self, cx, cy, w, h, alpha=0.22):
        self.glow((cx - w / 2, cy - h / 2 + 5, cx + w / 2, cy + h / 2 + 5), 4, SHADOW_C, alpha, 7)

    def block(self, cx, cy, w, h, grid, edge, alpha=1.0, scale=1.0):
        w, h = w * scale, h * scale
        if w < 1:
            return
        a = int(255 * alpha)
        rows, cols = len(grid), len(grid[0])
        cw, ch = w / cols, h / rows
        g = min(cw, ch) * 0.09
        x0, y0 = cx - w / 2, cy - h / 2
        for i, row in enumerate(grid):
            for j, c in enumerate(row):
                self.d.rectangle(self.box((x0 + j * cw + g, y0 + i * ch + g,
                                           x0 + (j + 1) * cw - g, y0 + (i + 1) * ch - g)), fill=c + (a,))
        if edge is not None:
            self.rrect((x0 - 1.5, y0 - 1.5, x0 + w + 1.5, y0 + h + 1.5), 3, edge=edge, width=2.2, alpha=alpha)

    def arrow(self, p0, p1, frac, alpha, color, width=3.5, head=13):
        if frac <= 0.01 or alpha <= 0.01:
            return
        a = int(255 * alpha)
        ex, ey = lerp(p0[0], p1[0], frac), lerp(p0[1], p1[1], frac)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        hl = min(head, n * frac)
        bx, by = ex - ux * hl, ey - uy * hl
        px, py = -uy, ux
        lw = self.L(width)
        self.d.line([self.P(*p0), self.P(bx + ux * 2, by + uy * 2)], fill=color + (a,), width=int(round(lw)))
        sx, sy = self.P(*p0)
        self.d.ellipse([sx - lw / 2, sy - lw / 2, sx + lw / 2, sy + lw / 2], fill=color + (a,))
        hw = hl * 0.55
        self.d.polygon([self.P(ex, ey), self.P(bx + px * hw, by + py * hw), self.P(bx - px * hw, by - py * hw)],
                       fill=color + (a,))

    def text(self, x, y, s, size, color):
        px = max(1, int(round(size * self.s * SS)))
        if px not in FONTS:
            FONTS[px] = ImageFont.truetype(FONT_PATH, px)
        self.d.text(self.P(x, y), s, font=FONTS[px], fill=color + (255,), anchor="mt")

    def final(self):
        return self.img.resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------- state
def sm_style(row, col, t):
    if row == 0:
        u = col
        st = blend_style(IDLE, GREEN, prog(t, ARRIVE[u], ARRIVE[u] + 0.2))
        if ARRIVE[u] <= t < FINISH[u]:
            st = st[:3] + (st[3] * (0.8 + 0.2 * math.sin(2 * math.pi * 1.6 * t)),)
        elif t >= FINISH[u]:
            st = st[:3] + (lerp(st[3], 0.3, prog(t, FINISH[u], LANDED[u])),)
        return blend_style(st, YELLOW, prog(t, LANDED[u], LANDED[u] + 0.25))
    if col >= 2:
        return GREY
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.1 * (t - T_RED[0]) - math.pi / 2)
    red = RED[:3] + (0.25 + 0.4 * pulse,)
    st = blend_style(IDLE, red, ease(prog(t, *T_RED)))
    green = GREEN[:3] + (GREEN[3] * (0.8 + 0.2 * math.sin(2 * math.pi * 1.6 * t)),)
    return blend_style(st, green, prog(t, READ_END[col], READ_END[col] + 0.25))


def render(t, fi):
    F = Frame(t)
    split = ease(prog(t, *T_SPLIT))

    # SM glows, gmem, SMs
    for row in range(2):
        for col in range(4):
            st = sm_style(row, col, t)
            F.glow(sm_box(row, col), 20, st[2], st[3], 14)
    F.rrect(GMEM, 16, fill=GMEM_FILL, edge=GMEM_EDGE, width=2.5)
    for row in range(2):
        for col in range(4):
            st = sm_style(row, col, t)
            F.rrect(sm_box(row, col), 20, fill=st[0], edge=st[1], width=3)
            b = sm_box(row, col)
            F.text((b[0] + b[2]) / 2, b[3] + 7, f"SM{row * 4 + col}", 16, LABEL_C)
    F.text((GMEM[0] + GMEM[2]) / 2, GMEM[3] + 9, "Global Memory", 18, LABEL_C)

    # ghosts of launched units, units still at home
    for u in range(4):
        hx, hy = unit_home(u, split)
        if t >= LAUNCH[u]:
            ga = ease(prog(t, LAUNCH[u], LAUNCH[u] + 0.4))
            F.rrect((hx - UW / 2, hy - UH / 2, hx + UW / 2, hy + UH / 2), 3, edge=GHOST, width=2, alpha=ga)
    for u in range(4):
        if t < LAUNCH[u]:
            hx, hy = unit_home(u, split)
            F.block(hx, hy, UW, UH, mix_grid(RAW[u], TINT[u], split), None)
            if split > 0.01:
                F.rrect((hx - UW / 2 - 1.5, hy - UH / 2 - 1.5, hx + UW / 2 + 1.5, hy + UH / 2 + 1.5), 3,
                        edge=UNIT_C[u], width=2.2, alpha=split)

    # counters
    for u in range(4):
        if t < FINISH[u]:
            continue
        bc = u // 2
        sc = back_out(prog(t, FINISH[u], FINISH[u] + 0.3))
        sc *= 1 - ease(prog(t, READ_START[bc], READ_START[bc] + 0.3))
        if sc <= 0.01:
            continue
        g = prog(t, LANDED[u], LANDED[u] + 0.25)
        cx, cy = ctr_center(u)
        half = CTR / 2 * sc
        if g > 0:
            F.glow((cx - half, cy - half, cx + half, cy + half), 7, CTR_DONE_FILL, 0.45 * g * sc, 6)
        F.rrect((cx - half, cy - half, cx + half, cy + half), 7 * sc,
                fill=mix(CTR_PENDING_FILL, CTR_DONE_FILL, g), edge=mix(CTR_PENDING_EDGE, CTR_DONE_EDGE, g),
                width=2.5)

    # arrows from waiting bottom SMs to their two counters
    for bc in range(2):
        frac = ease(prog(t, READY[bc], READY[bc] + ARROW_T))
        fade = 1 - prog(t, READ_START[bc] + 0.35, READ_START[bc] + 0.65)
        sx, sy = sm_center(1, bc)
        bottom = sm_box(1, bc)[3]
        for k in range(2):
            u = 2 * bc + k
            cx, cy = ctr_center(u)
            F.arrow((sx + (k - 0.5) * 56, bottom + 30), (cx, cy - CTR / 2 - 7), frac, fade, ARROW_C)

    # unit / data blocks
    movers = []
    for u in range(4):
        bc, k = u // 2, u % 2
        smx, smy = sm_center(0, u)
        if LAUNCH[u] <= t < ARRIVE[u]:
            e = ease(prog(t, LAUNCH[u], ARRIVE[u]))
            hx, hy = unit_home(u, 1.0)
            movers.append((lerp(hx, smx, e), lerp(hy, smy, e), lerp(UW, 140, e), lerp(UH, 35, e), TINT[u], u))
        elif ARRIVE[u] <= t < FINISH[u]:
            q = ease(prog(t, ARRIVE[u] + 0.55 * CRUNCH[u], FINISH[u]))
            grid = shimmer(mix_grid(TINT[u], POST[u], q), fi * 13 + u, 0.75 * (1 - q))
            F.block(smx, smy, lerp(140, DATA_W, q), lerp(35, DATA_H, q), grid, UNIT_C[u])
        elif FINISH[u] <= t < LANDED[u]:
            e = ease(prog(t, FINISH[u], LANDED[u]))
            dx, dy = data_center(u)
            movers.append((lerp(smx, dx, e), lerp(smy, dy, e), DATA_W, DATA_H, POST[u], u))
        elif LANDED[u] <= t < READ_START[bc]:
            dx, dy = data_center(u)
            F.block(dx, dy, DATA_W, DATA_H, POST[u], UNIT_C[u])
        elif READ_START[bc] <= t < READ_END[bc]:
            e = ease(prog(t, READ_START[bc], READ_END[bc]))
            dx, dy = data_center(u)
            bx, by = sm_center(1, bc)
            tx = bx + (k - 0.5) * 75
            movers.append((lerp(dx, tx, e), lerp(dy, by, e), lerp(DATA_W, 72, e), lerp(DATA_H, 24.4, e),
                           POST[u], u))
        elif t >= READ_END[bc]:
            bx, by = sm_center(1, bc)
            amp = 0.7 * ease(prog(t, READ_END[bc], READ_END[bc] + 0.3))
            F.block(bx + (k - 0.5) * 75, by, 72, 24.4, shimmer(POST[u], fi * 17 + u + 50, amp), UNIT_C[u])
    for cx, cy, w, h, grid, u in movers:
        F.shadow(cx, cy, w, h)
    for cx, cy, w, h, grid, u in movers:
        F.block(cx, cy, w, h, grid, UNIT_C[u])
    return F.final()


def main():
    out = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--frames":
        for ts in sys.argv[4:]:
            t = float(ts)
            render(t, int(t * FPS)).save(f"{sys.argv[3]}/f_{t:05.2f}.png")
        return
    n = int(round(T_END * FPS))
    frames = []
    first = None
    for fi in range(n):
        t = fi / FPS
        im = render(t, fi)
        if fi == 0:
            first = im
        lp = ease(prog(t, *T_LOOP))
        if lp > 0:
            im = Image.blend(im, first, lp)
        frames.append(im)
    # Global palette: exact white, plus 255 colours fit to the non-white pixels, each distinct colour
    # weighted by min(count, cap) so faint glow gradients get entries instead of being swamped by flat fills.
    px = np.concatenate([np.asarray(im).reshape(-1, 3) for im in frames[::5]])
    px = px[(px != 255).any(axis=1)]
    uniq, counts = np.unique(px, axis=0, return_counts=True)
    reps = np.minimum(counts, 40)
    sample_px = np.repeat(uniq, reps, axis=0)
    side = int(math.ceil(math.sqrt(len(sample_px))))
    buf = np.full((side * side, 3), sample_px[0], dtype=np.uint8)
    buf[:len(sample_px)] = sample_px
    fitted = Image.fromarray(buf.reshape(side, side, 3)).quantize(colors=255, method=Image.MEDIANCUT, kmeans=3)
    colours = fitted.getpalette()[:255 * 3] + [255, 255, 255]
    table = np.array(colours, dtype=np.int32).reshape(256, 3)
    pframes = []
    for im in frames:                  # exact nearest-colour mapping (Pillow's lookup is approximate)
        a = np.asarray(im).reshape(-1, 3)
        u, inv = np.unique(a, axis=0, return_inverse=True)
        d = ((u[:, None, :].astype(np.int32) - table[None, :, :]) ** 2).sum(-1)
        idx = d.argmin(1).astype(np.uint8)[inv].reshape(H, W)
        p = Image.fromarray(idx, "P")
        p.putpalette(colours)
        pframes.append(p)
    pframes[0].save(out, save_all=True, append_images=pframes[1:], duration=1000 // FPS, loop=0,
                    optimize=False, disposal=1)
    print("frames", n, "->", out)


if __name__ == "__main__":
    main()
