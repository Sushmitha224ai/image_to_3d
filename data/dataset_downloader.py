import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def generate_synthetic_paired_dataset(output_dir="data/raw", num_identities=100, img_size=(256, 256)):
    """
    Generates a structured, paired forensic sketch-photo benchmark dataset with 
    distinct facial identities, feature variations, and line-art styles.
    Each subject has a unique identity ID to guarantee NO identity leakage between splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    sketch_dir = os.path.join(output_dir, "sketches")
    photo_dir = os.path.join(output_dir, "photos")
    os.makedirs(sketch_dir, exist_ok=True)
    os.makedirs(photo_dir, exist_ok=True)

    print(f"[*] Generating {num_identities} unique paired forensic sketch-photo identities in '{output_dir}'...")

    np.random.seed(42)

    for i in range(1, num_identities + 1):
        id_str = f"id_{i:04d}"

        # Randomize facial features for identity uniqueness
        skin_tone = (
            np.random.randint(180, 245),
            np.random.randint(140, 205),
            np.random.randint(110, 175)
        )
        hair_color = (
            np.random.randint(20, 80),
            np.random.randint(20, 60),
            np.random.randint(20, 50)
        )
        eye_color = (
            np.random.randint(40, 100),
            np.random.randint(30, 70),
            np.random.randint(20, 60)
        )
        
        # Face geometry parameters
        face_width = np.random.randint(130, 170)
        face_height = np.random.randint(160, 200)
        center_x, center_y = 128, 132

        # --- 1. Create Photorealistic 2D Color Face ---
        photo = Image.new("RGB", img_size, (230, 235, 240))
        draw = ImageDraw.Draw(photo)

        # Draw Neck
        draw.rectangle([center_x - 30, center_y + 40, center_x + 30, center_y + 100], fill=(skin_tone[0]-20, skin_tone[1]-20, skin_tone[2]-20))

        # Draw Face Ellipse
        face_box = [center_x - face_width // 2, center_y - face_height // 2, center_x + face_width // 2, center_y + face_height // 2]
        draw.ellipse(face_box, fill=skin_tone, outline=(skin_tone[0]-40, skin_tone[1]-40, skin_tone[2]-40), width=2)

        # Draw Eyes
        eye_offset_x = np.random.randint(25, 35)
        eye_y = center_y - 20
        eye_r = np.random.randint(8, 14)
        
        # Left Eye
        draw.ellipse([center_x - eye_offset_x - eye_r, eye_y - eye_r, center_x - eye_offset_x + eye_r, eye_y + eye_r], fill=(255, 255, 255), outline=(50, 50, 50))
        draw.ellipse([center_x - eye_offset_x - eye_r//2, eye_y - eye_r//2, center_x - eye_offset_x + eye_r//2, eye_y + eye_r//2], fill=eye_color)
        
        # Right Eye
        draw.ellipse([center_x + eye_offset_x - eye_r, eye_y - eye_r, center_x + eye_offset_x + eye_r, eye_y + eye_r], fill=(255, 255, 255), outline=(50, 50, 50))
        draw.ellipse([center_x + eye_offset_x - eye_r//2, eye_y - eye_r//2, center_x + eye_offset_x + eye_r//2, eye_y + eye_r//2], fill=eye_color)

        # Eyebrows
        draw.line([center_x - eye_offset_x - eye_r - 5, eye_y - 15, center_x - eye_offset_x + eye_r + 5, eye_y - 18], fill=hair_color, width=4)
        draw.line([center_x + eye_offset_x - eye_r - 5, eye_y - 18, center_x + eye_offset_x + eye_r + 5, eye_y - 15], fill=hair_color, width=4)

        # Nose
        nose_y1, nose_y2 = eye_y + 5, center_y + 15
        draw.line([center_x, nose_y1, center_x - 8, nose_y2, center_x + 8, nose_y2], fill=(skin_tone[0]-35, skin_tone[1]-35, skin_tone[2]-35), width=2)

        # Mouth
        mouth_y = center_y + 40
        mouth_w = np.random.randint(25, 38)
        draw.ellipse([center_x - mouth_w, mouth_y - 6, center_x + mouth_w, mouth_y + 6], fill=(190, 80, 80), outline=(130, 40, 40))

        # Hair
        hair_style = i % 3
        if hair_style == 0: # Short Hair
            draw.arc([center_x - face_width // 2 - 10, center_y - face_height // 2 - 20, center_x + face_width // 2 + 10, center_y], 180, 360, fill=hair_color, width=25)
        elif hair_style == 1: # Long Hair
            draw.rectangle([center_x - face_width // 2 - 15, center_y - face_height // 2 - 20, center_x - face_width // 2 + 15, center_y + 80], fill=hair_color)
            draw.rectangle([center_x + face_width // 2 - 15, center_y - face_height // 2 - 20, center_x + face_width // 2 + 15, center_y + 80], fill=hair_color)
            draw.ellipse([center_x - face_width // 2 - 10, center_y - face_height // 2 - 25, center_x + face_width // 2 + 10, center_y], fill=hair_color)
        else: # Curly/Wavy
            for h_offset in range(-face_width//2, face_width//2, 15):
                draw.ellipse([center_x + h_offset - 10, center_y - face_height//2 - 25, center_x + h_offset + 15, center_y - face_height//2 + 15], fill=hair_color)

        # Smooth photo with slight Gaussian blur
        photo = photo.filter(ImageFilter.SMOOTH)

        # --- 2. Create Forensic Sketch Version ---
        # Convert photo to grayscale
        gray = photo.convert("L")
        
        # Extract edge map using pencil sketch style (Laplacian/Sobel/Dodge blend)
        inv_gray = Image.eval(gray, lambda x: 255 - x)
        blurred_inv = inv_gray.filter(ImageFilter.GaussianBlur(radius=int(np.random.choice([2, 3, 5]))))
        
        # Color dodge blending for realistic pencil sketch texture
        sketch_arr = np.array(gray, dtype=float)
        blur_arr = np.array(blurred_inv, dtype=float)
        
        # Avoid division by zero
        dodge_arr = np.minimum(255, (sketch_arr * 256.0) / (255.0 - blur_arr + 1e-5))
        sketch_img = Image.fromarray(np.uint8(dodge_arr))

        # Add artistic sketch noise / texture variation per identity
        if i % 4 == 0:
            # High-contrast pencil sketch
            enhancer = ImageEnhance.Contrast(sketch_img)
            sketch_img = enhancer.enhance(1.8)
        elif i % 4 == 1:
            # Soft charcoal sketch
            sketch_img = sketch_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        elif i % 4 == 2:
            # Fine line-art sketch
            enhancer = ImageEnhance.Sharpness(sketch_img)
            sketch_img = enhancer.enhance(2.5)

        # Save identity files
        sketch_path = os.path.join(sketch_dir, f"{id_str}_sketch.png")
        photo_path = os.path.join(photo_dir, f"{id_str}_photo.png")

        sketch_img.convert("RGB").save(sketch_path)
        photo.save(photo_path)

    print(f"[OK] Successfully generated {num_identities} sketch-photo pairs in '{output_dir}'.")
    return output_dir

if __name__ == "__main__":
    generate_synthetic_paired_dataset()
