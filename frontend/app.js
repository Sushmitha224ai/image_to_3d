let scene, camera, renderer, controls, faceMesh;

const MODEL_NAMES = {
  "model_1_pix2pix": "Pix2Pix Conditional GAN",
  "model_2_unet": "Deep U-Net (Perceptual Loss)",
  "model_3_resnet_gan": "ResNet Bottleneck LSGAN",
  "model_4_pca_ridge": "Eigenface Dual-PCA + Ridge",
  "model_5_knn_patch": "Multi-Scale Patch K-NN"
};

document.addEventListener("DOMContentLoaded", () => {
  initUI();
  try {
    init3DViewer();
  } catch (err) {
    console.warn("3D Viewer Initialization warning:", err);
  }
  loadDefaultSample();
});

// Auto-preload sample sketch so the page is ready to run immediately upon opening
async function loadDefaultSample() {
  try {
    const res = await fetch("/static/sample_sketch.png");
    if (res.ok) {
      const blob = await res.blob();
      const defaultFile = new File([blob], "sample_sketch.png", { type: "image/png" });
      if (typeof handleFileSelect === "function") {
        handleFileSelect(defaultFile);
      }
    }
  } catch (err) {
    console.log("Preload default sample info:", err);
  }
}

function initUI() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  if (fileInput) {
    fileInput.addEventListener("click", () => {
      fileInput.value = "";
    });

    const onFileChange = (e) => {
      if (e.target.files && e.target.files.length > 0) {
        if (typeof handleFileSelect === "function") {
          handleFileSelect(e.target.files[0]);
        }
      }
    };

    fileInput.addEventListener("change", onFileChange);
    fileInput.addEventListener("input", onFileChange);
  }

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        if (typeof handleFileSelect === "function") {
          handleFileSelect(e.dataTransfer.files[0]);
        }
      }
    });
  }
}

window.renderResults = function(data) {
  // 1. Render Selected Model Header Badge
  const selectedBadge = document.getElementById("selectedBadge");
  if (selectedBadge) {
    selectedBadge.innerText = `SELECTED MODEL: ${data.selected_model_name.toUpperCase()}`;
    selectedBadge.style.display = "inline-block";
  }

  // 2. Render Model Scores List
  const modelScoreList = document.getElementById("modelScoreList");
  if (modelScoreList) {
    modelScoreList.innerHTML = "";
    const selectedKey = data.selected_model_key;

    Object.keys(MODEL_NAMES).forEach(key => {
      const isSelected = (key === selectedKey);
      const scoreVal = (data.model_scores[key] || 0.70) * 100;
      const probVal = (data.model_probabilities[key] || 0.20) * 100;

      const row = document.createElement("div");
      row.className = `score-row ${isSelected ? "selected" : ""}`;
      row.innerHTML = `
        <div class="score-info">
          <span style="font-weight:600;">${MODEL_NAMES[key]} ${isSelected ? "← SELECTED" : ""}</span>
          <span style="font-weight:600;">${probVal.toFixed(1)}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${probVal}%;"></div>
        </div>
      `;
      modelScoreList.appendChild(row);
    });
  }

  // 3. Render Selected 2D Face Output
  const container2D = document.getElementById("photo2DPreviewContainer");
  if (container2D) {
    container2D.innerHTML = `<img src="data:image/png;base64,${data.generated_2d_b64}" alt="Generated 2D Face" style="max-width:100%; max-height:100%; object-fit:contain;">`;
  }
  const label2D = document.getElementById("selectedModelLabel");
  if (label2D) {
    label2D.innerText = `Automatically Selected: ${data.selected_model_name} (${data.confidence_percent}% confidence)`;
  }

  // 4. Render All Candidate Previews Grid
  const candidatesGrid = document.getElementById("candidatesGrid");
  if (candidatesGrid) {
    candidatesGrid.innerHTML = "";
    const selectedKey = data.selected_model_key;
    Object.keys(data.candidate_previews).forEach(key => {
      const isSel = (key === selectedKey);
      const card = document.createElement("div");
      card.style.cssText = `
        background: rgba(255,255,255,0.03);
        border: 1px solid ${isSel ? '#10b981' : 'rgba(255,255,255,0.08)'};
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
      `;
      card.innerHTML = `
        <img src="data:image/png;base64,${data.candidate_previews[key]}" style="width:100%; height:100px; object-fit:cover; border-radius:4px;">
        <p style="font-size:0.7rem; color:${isSel ? '#10b981' : '#94a3b8'}; margin-top:0.35rem; font-weight:${isSel?'700':'400'};">
          ${MODEL_NAMES[key]}
        </p>
      `;
      candidatesGrid.appendChild(card);
    });
  }

  // 5. Update 3D Face Representation in Three.js Scene
  update3DFaceMesh(data.generated_2d_b64);
};

/* ----------------------------------------------------
   Three.js Interactive 3D WebGL Viewer Engine
---------------------------------------------------- */
window.init3DViewer = function() {
  const container = document.getElementById("viewer3DContainer");
  if (!container) return;
  const width = container.clientWidth || 400;
  const height = container.clientHeight || 380;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050811);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 0, 3.5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  if (typeof THREE.OrbitControls !== 'undefined') {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
  }

  // Lighting Setup
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(2, 2, 4);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 0.4);
  dirLight2.position.set(-2, -1, -2);
  scene.add(dirLight2);

  // Initial Placeholder 3D Face Geometry
  createPlaceholderMesh();

  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  animate();
};

function createPlaceholderMesh() {
  if (faceMesh) scene.remove(faceMesh);

  const res = 32;
  const geom = new THREE.PlaneGeometry(1.6, 2.0, res, res);
  const pos = geom.attributes.position;
  const uv = geom.attributes.uv;

  for (let i = 0; i < pos.count; i++) {
    const u = uv.getX(i);
    const v = uv.getY(i);

    const norm_x = (u - 0.5) * 2.0;
    const norm_y = (v - 0.5) * 2.0;
    const dist_sq = norm_x**2 + norm_y**2;

    const z = Math.sqrt(Math.max(0, 1.0 - dist_sq * 0.75)) * 0.45;
    pos.setZ(i, z);
  }

  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    wireframe: true,
    roughness: 0.4
  });

  faceMesh = new THREE.Mesh(geom, mat);
  scene.add(faceMesh);
}

window.update3DFaceMesh = function(textureB64) {
  if (!scene) return;
  if (faceMesh) scene.remove(faceMesh);

  const loader = new THREE.TextureLoader();
  const texture = loader.load("data:image/png;base64," + textureB64);
  texture.flipY = false;

  const res = 48;
  const geom = new THREE.PlaneGeometry(1.6, 2.0, res, res);

  const pos = geom.attributes.position;
  const uv = geom.attributes.uv;

  for (let i = 0; i < pos.count; i++) {
    const u = uv.getX(i);
    const v = uv.getY(i);

    const norm_x = (u - 0.5) * 2.0;
    const norm_y = (v - 0.5) * 2.0;
    const dist_sq = norm_x**2 + norm_y**2;

    const z_base = Math.sqrt(Math.max(0, 1.0 - dist_sq * 0.75)) * 0.45;
    const dist_nose = Math.sqrt((u - 0.5)**2 + (v - 0.55)**2);
    const z_nose = 0.18 * Math.exp(-dist_nose**2 / 0.02);

    const dist_leye = Math.sqrt((u - 0.32)**2 + (v - 0.68)**2);
    const dist_reye = Math.sqrt((u - 0.68)**2 + (v - 0.68)**2);
    const z_eyes = -0.08 * (Math.exp(-dist_leye**2 / 0.01) + Math.exp(-dist_reye**2 / 0.01));

    const z = (z_base + z_nose + z_eyes) * (dist_sq <= 1.2 ? 1 : 0);
    pos.setZ(i, z);
  }

  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.5,
    metalness: 0.1,
    side: THREE.DoubleSide
  });

  faceMesh = new THREE.Mesh(geom, mat);
  scene.add(faceMesh);
  resetCamera();
};

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

window.setCameraPreset = function(preset) {
  if (!camera) return;
  switch (preset) {
    case 'front':
      camera.position.set(0, 0, 3.5);
      break;
    case 'left':
      camera.position.set(-3.0, 0, 1.5);
      break;
    case 'right':
      camera.position.set(3.0, 0, 1.5);
      break;
    case 'top':
      camera.position.set(0, 3.5, 0.5);
      break;
  }
  if (controls) {
    controls.target.set(0, 0, 0);
    controls.update();
  }
};

window.resetCamera = function() {
  setCameraPreset('front');
};
