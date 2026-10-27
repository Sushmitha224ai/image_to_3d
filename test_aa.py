import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

# test antialiased
n = 128
x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-2.0, 2.0, n)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

ls = LightSource(azdeg=315, altdeg=55)
rgb = ls.shade(Z, cmap=plt.cm.copper, blend_mode="soft")
if rgb.shape[2] == 4: rgb = rgb[:, :, :3]

mask_x = np.cos(np.clip(X / 1.5 * (np.pi / 2), -np.pi/2, np.pi/2))
mask_y = np.cos(np.clip(Y / 2.0 * (np.pi / 2), -np.pi/2, np.pi/2))
alpha = (mask_x * mask_y)**0.5
rgba = np.dstack((rgb, alpha))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=rgba, rstride=1, cstride=1, antialiased=False, shade=False)
ax.view_init(elev=70, azim=-90)
ax.set_axis_off()
plt.savefig("test_aa.png", transparent=True, bbox_inches="tight")
