import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from PIL import Image
import io

n = 128
x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-2.0, 2.0, n)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

ls = LightSource(azdeg=315, altdeg=55)
rgb = ls.shade(Z, cmap=plt.cm.copper, blend_mode="soft")
if rgb.shape[2] == 4: rgb = rgb[:, :, :3]

# Fade to pure black at the borders
mask_x = np.cos(np.clip(X / 1.5 * (np.pi / 2), -np.pi/2, np.pi/2))
mask_y = np.cos(np.clip(Y / 2.0 * (np.pi / 2), -np.pi/2, np.pi/2))
mask_3d = ((mask_x * mask_y)**0.5)[:, :, np.newaxis]
rgb = rgb * mask_3d

# Plot with pure black background
fig = plt.figure(figsize=(6, 6), facecolor="black")
ax = fig.add_subplot(111, projection='3d', facecolor="black")
ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, antialiased=True, shade=False)
ax.view_init(elev=70, azim=-90)
ax.set_axis_off()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format="png", facecolor="black", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
buf.seek(0)

# Load into PIL
img = Image.open(buf).convert("RGB")
# Create an alpha channel from the grayscale lightness of the image!
# Pure black background becomes 0 (transparent).
grayscale = img.convert("L")
# Boost the alpha a bit so the face doesn't become too transparent
alpha_data = np.array(grayscale)
alpha_data = np.clip(alpha_data * 2.5, 0, 255).astype(np.uint8)

img.putalpha(Image.fromarray(alpha_data))
img.save("test_lightness_alpha.png")
