import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from PIL import Image, ImageDraw, ImageFilter
import io

n = 128
x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-2.0, 2.0, n)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

ls = LightSource(azdeg=315, altdeg=55)
rgb = ls.shade(Z, cmap=plt.cm.copper, blend_mode="soft")
if rgb.shape[2] == 4: rgb = rgb[:, :, :3]

# No 3D masking here! Just plot the surface.
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, antialiased=True, shade=False)
ax.view_init(elev=70, azim=-90)
ax.set_axis_off()

# Save to buffer with transparent background
buf = io.BytesIO()
plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
buf.seek(0)

# Load into PIL
img = Image.open(buf).convert("RGBA")
width, height = img.size

# Create a 2D radial gradient mask
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)
# Draw an ellipse that fills most of the image, fading out
# We want the center to be 255 (opaque), edges 0 (transparent)
bbox = [0, 0, width, height]
draw.ellipse(bbox, fill=255)

# Blur the mask heavily for a soft radial fade
mask = mask.filter(ImageFilter.GaussianBlur(radius=width//6))

# Apply the mask to the alpha channel of the image
# First, get the original alpha
r, g, b, a = img.split()
# Multiply original alpha by our soft mask
# out_alpha = a * (mask / 255)
out_alpha = np.array(a).astype(float) * (np.array(mask).astype(float) / 255.0)
new_alpha = Image.fromarray(out_alpha.astype(np.uint8), mode="L")
img.putalpha(new_alpha)

img.save("test_pil_mask.png")
