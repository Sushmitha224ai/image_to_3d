import requests
import base64
import io
from PIL import Image, ImageDraw

# Create a synthetic sketch
img = Image.new("L", (256, 256), 220)
draw = ImageDraw.Draw(img)
draw.ellipse([50, 30, 206, 230], outline=50, width=3)
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

print("Sending request to backend on port 8095...")
try:
    resp = requests.post(
        "http://127.0.0.1:8095/predict-3d/",
        files={"file": ("test_sketch.png", buf, "image/png")},
        timeout=10,
    )
    if resp.status_code == 200:
        res = resp.json()
        print("Success! Selected model:", res["selected_model"])
        img_bytes = base64.b64decode(res["image_base64"])
        with open("test_transparency.png", "wb") as f:
            f.write(img_bytes)
        print("Saved 'test_transparency.png'")
    else:
        print("Failed:", resp.status_code, resp.text)
except Exception as e:
    print("Error:", e)
