"""Generate assets/hero.jpg: a stylised SAR interferogram over Earth's limb (procedural, no external art)."""
import numpy as np
from PIL import Image

W, H = 1920, 460
rng = np.random.default_rng(7)
y, x = np.mgrid[0:H, 0:W].astype(np.float64)
img = np.zeros((H, W, 3))

# Deep-space vertical gradient
t = y / H
img[...] = (np.array([3, 8, 22]) * (1 - t)[..., None] + np.array([9, 27, 66]) * t[..., None])

# Stars
for _ in range(520):
    sx, sy = rng.integers(0, W), rng.integers(0, int(H * 0.8))
    b = rng.uniform(80, 255)
    img[sy, sx] = np.minimum(255, img[sy, sx] + b)
    if rng.random() < 0.08:
        img[max(sy - 1, 0):sy + 2, max(sx - 1, 0):sx + 2] += b * 0.35

# Earth limb: large circle below the frame with an atmospheric glow
cx, cy, R = W * 0.5, H + 2650, 2900
d = np.hypot(x - cx, y - cy)
earth = d < R
glow = np.exp(-np.clip(d - R, 0, None) / 38.0) * (d >= R)
img += glow[..., None] * np.array([40, 120, 255]) * 0.9
shade = np.clip((R - d) / 260.0, 0, 1)
surface = np.array([6, 18, 44]) + shade[..., None] * np.array([10, 30, 60])
img = np.where(earth[..., None], surface, img)

# Interferometric fringes (wrapped phase of a double deformation bowl) draped over the surface
fx, fy = x - W * 0.70, y - H * 0.86
phase = 0.0
for ax, ay, amp, s in [(0, 0, 9.0, 330.0), (-210, 30, 5.0, 210.0)]:
    phase += amp * np.exp(-(((fx - ax) / s) ** 2 + ((fy - ay) / (s * 0.55)) ** 2))
wrapped = (phase * 2 * np.pi) % (2 * np.pi)
r = 0.5 + 0.5 * np.cos(wrapped + 0.0)
g = 0.5 + 0.5 * np.cos(wrapped + 2.1)
b = 0.5 + 0.5 * np.cos(wrapped + 4.2)
fringe = (np.stack([r, g, b], axis=-1) * 0.62 + 0.14 * np.array([0.3, 0.5, 1.0])) * 255
mask = earth.astype(float) * np.clip((phase - 0.25) / 1.6, 0, 1) ** 1.5 * 0.8
img = img * (1 - mask[..., None]) + fringe * mask[..., None]

# Faint radar swath lines and a satellite-track arc
for k in range(7):
    dd = np.abs(np.hypot(x - W * 0.2, y + 900) - (1050 + k * 55))
    img += (np.exp(-(dd / 1.4) ** 2) * (55 - k * 6))[..., None] * np.array([0.25, 0.55, 1.0])
img += np.exp(-(((x - W * 0.24) ** 2 + (y - H * 0.2) ** 2) / 14.0))[..., None] * np.array([200, 230, 255]) * 0.9

# Left-to-right darkening so the title stays readable
img *= (0.45 + 0.55 * np.clip(x / (W * 0.65), 0, 1))[..., None] ** 1.0 * 0 + (0.38 + 0.62 * np.clip(x / (W * 0.62), 0, 1))[..., None]

Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save("assets/hero.jpg", quality=88)
print("ok")

