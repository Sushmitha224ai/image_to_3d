"""
=============================================================================
 app_frontend.py — Streamlit Web App for Forensic 3D Face Reconstruction
=============================================================================

Architecture Overview
─────────────────────
  User Browser (Streamlit UI)
        │
        ├── 1. File Upload Widget (JPG/PNG sketch)
        │
        ├── 2. POST request to FastAPI backend (/predict-3d/)
        │       └── Sends sketch as multipart/form-data
        │
        ├── 3. Receives JSON response:
        │       ├── selected_model (str)
        │       ├── confidence (float)
        │       ├── profile (dict)
        │       ├── all_scores (dict)
        │       └── image_base64 (str)
        │
        ├── 4. Displays split layout:
        │       ├── Left Column:  Original 2D sketch
        │       └── Right Column: 3D rendered face + model badge
        │
        └── 5. Download button for the 3D face PNG

Run with:
    streamlit run app_frontend.py
=============================================================================
"""

import io
import base64
import requests
import streamlit as st

# =============================================================================
#  Configuration
# =============================================================================

# FastAPI backend URL (default: localhost:8000)
BACKEND_URL = "http://localhost:8095"
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict-3d/"

# =============================================================================
#  Page Configuration & Custom Styling
# =============================================================================

st.set_page_config(
    page_title="ForensicFace3D — AI Sketch-to-3D Reconstruction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject custom CSS for a high-end forensic UI ────────────────────────────
st.markdown("""
<style>
    /* ── Global dark theme ─────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(145deg, #0a0e17 0%, #0d1321 40%, #111827 100%);
    }

    /* ── Header styling ────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-family: 'Courier New', monospace;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #64748b;
        font-size: 0.95rem;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 1px;
    }

    /* ── Card containers ───────────────────────────────────────────────── */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* ── Model selection badge ─────────────────────────────────────────── */
    .model-badge {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .model-badge .model-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
    }
    .model-badge .confidence {
        font-size: 0.9rem;
        color: #a78bfa;
        margin-top: 0.3rem;
    }

    /* ── Profile metric pills ──────────────────────────────────────────── */
    .metric-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 0.8rem 0;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .metric-pill .value {
        color: #38bdf8;
        font-weight: 700;
    }

    /* ── Score bar chart ───────────────────────────────────────────────── */
    .score-bar-container {
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .score-label {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        min-width: 145px;
        text-align: right;
    }
    .score-bar-bg {
        flex: 1;
        height: 14px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 7px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 7px;
        transition: width 0.6s ease;
    }
    .score-bar-fill.dl {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }
    .score-bar-fill.ml {
        background: linear-gradient(90deg, #34d399, #a3e635);
    }
    .score-value {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        color: #64748b;
        min-width: 48px;
    }

    /* ── Section titles ────────────────────────────────────────────────── */
    .section-title {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(100, 116, 139, 0.2);
    }

    /* ── Footer ────────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #334155;
        font-size: 0.75rem;
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
    }

    /* ── Hide Streamlit default elements for cleaner look ───────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  1. Header
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🧬 FORENSICFACE3D</h1>
    <p>AUTOMATED 3D HUMAN FACE RECONSTRUCTION FROM FORENSIC SKETCHES</p>
    <p style="font-size: 0.8rem; color: #334155; margin-top: 0.5rem;">
        AI-Powered  •  7-Model Smart Router  •  Volumetric Rendering Pipeline
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
#  2. File Upload Widget
# =============================================================================

st.markdown('<p class="section-title">📁 Upload Forensic Sketch</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a forensic sketch (JPG/PNG) to generate a 3D face reconstruction",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Max file size: 200MB.",
    label_visibility="collapsed",
)

# =============================================================================
#  3. Processing Pipeline — On Upload
# =============================================================================

if uploaded_file is not None:
    # ── Read uploaded bytes for display and transmission ─────────────────
    file_bytes = uploaded_file.getvalue()

    # ── Split layout: Left (sketch) | Right (3D output) ─────────────────
    col_left, col_spacer, col_right = st.columns([1, 0.05, 1])

    # ────────────────────────────────────────────────────────────────────
    #  LEFT COLUMN — Original 2D Sketch Display
    # ────────────────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<p class="section-title">🖼️ Input Forensic Sketch</p>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(
            file_bytes,
            caption=f"📄 {uploaded_file.name} ({len(file_bytes) / 1024:.1f} KB)",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    #  RIGHT COLUMN — Processing & 3D Output
    # ────────────────────────────────────────────────────────────────────
    with col_right:
        st.markdown('<p class="section-title">🧠 AI Model Router & 3D Output</p>', unsafe_allow_html=True)

        # ── Show processing spinner while backend works ──────────────────
        with st.spinner("🔄 Analyzing sketch and routing to optimal model..."):
            try:
                # ── Send sketch to FastAPI backend ───────────────────────
                # Data Flow: file_bytes → multipart POST → /predict-3d/
                response = requests.post(
                    PREDICT_ENDPOINT,
                    files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                    timeout=60,
                )

                if response.status_code != 200:
                    st.error(f"❌ Backend error (HTTP {response.status_code}): {response.text}")
                    st.stop()

                # ── Parse JSON response ──────────────────────────────────
                result = response.json()
                selected_model = result["selected_model"]
                confidence = result["confidence"]
                profile = result["profile"]
                all_scores = result["all_scores"]
                image_base64 = result["image_base64"]

                # ── Decode base64 image ──────────────────────────────────
                image_bytes = base64.b64decode(image_base64)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Cannot connect to backend.**\n\n"
                    "Please ensure the FastAPI server is running:\n"
                    "```\nuvicorn app_backend:app --reload --port 8000\n```"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The backend may be under heavy load.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.stop()

        # ── SUCCESS: Display results ─────────────────────────────────────

        # ── Model Selection Badge ────────────────────────────────────────
        # Dynamic status banner showing the automatically selected model
        st.markdown(f"""
        <div class="model-badge">
            <div class="model-name">🎯 Optimal Model Selected: {selected_model}</div>
            <div class="confidence">Confidence Score: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Sketch Profile Metrics ───────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-pill">Line Density: <span class="value">{profile['line_density']}</span></span>
            <span class="metric-pill">Contrast: <span class="value">{profile['contrast']}</span></span>
            <span class="metric-pill">Shading: <span class="value">{profile['shading']}</span></span>
        </div>
        """, unsafe_allow_html=True)

        # ── 3D Face Image Display ────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(
            image_bytes,
            caption=f"🧊 3D Volumetric Face — {selected_model}",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Download Button ──────────────────────────────────────────────
        # Allows users to download the 3D rendered face image directly
        st.download_button(
            label="⬇️  Download 3D Face Image",
            data=image_bytes,
            file_name=f"forensic_face_3d_{selected_model.lower().replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    #  Model Comparison Scoreboard (Below the split layout)
    # ════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown('<p class="section-title">📊 All Model Scores — Smart Router Breakdown</p>', unsafe_allow_html=True)

    # ── Deep Learning vs Machine Learning separation ─────────────────────
    dl_models = ["Pix2Pix GAN", "U-Net++", "Vision Transformer", "Volumetric CNN"]
    ml_models = ["SVR", "Random Forest", "MLP Regressor"]

    # Sort scores descending for visual ranking
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    max_score = max(all_scores.values())

    # ── Build HTML score bars ────────────────────────────────────────────
    bars_html = ""
    for model_name, score in sorted_scores:
        bar_width = (score / max_score) * 100
        css_class = "dl" if model_name in dl_models else "ml"
        is_selected = "⭐ " if model_name == selected_model else ""
        bars_html += f"""
        <div class="score-bar-container">
            <span class="score-label">{is_selected}{model_name}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill {css_class}" style="width: {bar_width}%"></div>
            </div>
            <span class="score-value">{score}%</span>
        </div>
        """

    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; gap: 1.5rem; margin-bottom: 0.8rem; justify-content: center;">
            <span style="font-size: 0.75rem; color: #818cf8;">■ Deep Learning</span>
            <span style="font-size: 0.75rem; color: #34d399;">■ Machine Learning</span>
        </div>
        {bars_html}
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state — guide the user ─────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</p>
        <p style="color: #64748b; font-family: 'Courier New', monospace; font-size: 0.9rem;">
            Upload a forensic sketch above to begin 3D face reconstruction.
        </p>
        <p style="color: #334155; font-size: 0.8rem; margin-top: 1rem;">
            The AI Smart Router will automatically evaluate 7 models<br>
            (4 Deep Learning + 3 Machine Learning) and select the optimal one.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
#  Footer
# =============================================================================

st.markdown("""
<div class="footer">
    FORENSICFACE3D  •  7th SEMESTER MAJOR PROJECT  •  COMPUTER SCIENCE ENGINEERING
    <br>
    POWERED BY FASTAPI + STREAMLIT + MATPLOTLIB  •  7-MODEL SMART ROUTER ARCHITECTURE
</div>
""", unsafe_allow_html=True)
