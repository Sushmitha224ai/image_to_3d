import os
import trimesh
import numpy as np
from PIL import Image

class Face3DExporter:
    """
    Exports reconstructed 3D Face Meshes with UV Texture maps into standard 3D formats:
    - Wavefront OBJ (.obj + .mtl + texture.png)
    - Polygon File Format PLY (.ply)
    - GL Transmission Format GLTF/GLB (.gltf / .glb)
    """
    def __init__(self, output_dir="output/3d"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_all(self, vertices, faces, uvs, texture_img, base_filename="reconstructed_face_3d"):
        """
        vertices: (N, 3) float array
        faces: (F, 3) int array
        uvs: (N, 2) float array
        texture_img: PIL Image or numpy array (H, W, 3)
        base_filename: base name without extension
        Returns dict containing absolute file paths of exported 3D formats.
        """
        if isinstance(texture_img, np.ndarray):
            if texture_img.dtype != np.uint8:
                if texture_img.min() < 0:
                    texture_img = ((texture_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                else:
                    texture_img = (texture_img * 255.0).clip(0, 255).astype(np.uint8)
            texture_pil = Image.fromarray(texture_img)
        else:
            texture_pil = texture_img

        # Save texture image file
        tex_path = os.path.join(self.output_dir, f"{base_filename}_texture.png")
        texture_pil.save(tex_path)

        # Create Trimesh geometry object with visual material texture
        material = trimesh.visual.texture.SimpleMaterial(image=texture_pil)
        visuals = trimesh.visual.TextureVisuals(uv=uvs, image=texture_pil, material=material)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, validate=True)

        exported_paths = {}

        # 1. Export OBJ
        obj_path = os.path.join(self.output_dir, f"{base_filename}.obj")
        mesh.export(obj_path, file_type="obj")
        exported_paths["obj"] = os.path.abspath(obj_path)

        # 2. Export PLY
        ply_path = os.path.join(self.output_dir, f"{base_filename}.ply")
        mesh.export(ply_path, file_type="ply")
        exported_paths["ply"] = os.path.abspath(ply_path)

        # 3. Export GLTF / GLB
        glb_path = os.path.join(self.output_dir, f"{base_filename}.glb")
        mesh.export(glb_path, file_type="glb")
        exported_paths["glb"] = os.path.abspath(glb_path)

        exported_paths["texture"] = os.path.abspath(tex_path)
        print(f"[OK] 3D Face successfully exported to '{self.output_dir}' (OBJ, PLY, GLB).")
        return exported_paths
