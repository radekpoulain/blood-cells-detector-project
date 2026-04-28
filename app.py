"""
Blood Cell Detector — Streamlit Web Application
Detects and classifies cells in peripheral blood smear images
using a YOLO26 model fine-tuned on 7 cell types.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
TEST_IMAGES_DIR = ROOT / "test_images"
METADATA_PATH = ROOT / "blood_detector_metadata.json"

# ── Constants ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "RBC",
    1: "Platelets",
    2: "Neutrophil",
    3: "Lymphocyte",
    4: "Monocyte",
    5: "Eosinophil",
    6: "Basophil",
}

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# Colors in RGB for Pillow drawing
COLORS = {
    "RBC":        (230, 57, 70),    # crimson red
    "Platelets":  (42, 157, 143),   # teal green
    "Neutrophil": (69, 123, 157),   # steel blue
    "Lymphocyte": (114, 93, 189),   # purple
    "Monocyte":   (244, 162, 97),   # sandy orange
    "Eosinophil": (233, 196, 106),  # gold
    "Basophil":   (38, 70, 83),     # dark teal
}

# Category grouping for summary
CATEGORY_ORDER = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                  "Monocyte", "Eosinophil", "Basophil"]

CATEGORY_EMOJI = {
    "RBC":        "🔴",
    "Platelets":  "🟢",
    "Neutrophil": "🔵",
    "Lymphocyte": "🟣",
    "Monocyte":   "🟠",
    "Eosinophil": "🟡",
    "Basophil":   "⚫",
}


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1d29 0%, #0e1117 100%);
    }

    /* Header styling */
    .main-header {
        background: rgba(26, 29, 41, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(230, 57, 70, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(230, 57, 70, 0.05) 0%, transparent 50%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .main-header h1 {
        background: linear-gradient(90deg, #E63946, #F4A261, #E63946);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        animation: shimmer 4s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 200% center; }
    }

    .main-header p {
        color: #a0a0b0;
        font-size: 1.1rem;
        margin: 0;
        position: relative;
        z-index: 1;
        max-width: 800px;
    }

    /* Stat cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .stat-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(230, 57, 70, 0.3);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #E63946;
        line-height: 1;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(230, 57, 70, 0.3);
    }

    .stat-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Detection table */
    .detection-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255, 255, 255, 0.02);
    }

    .detection-table th {
        background: rgba(26, 29, 41, 0.8);
        padding: 15px 20px;
        text-align: left;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #E63946;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .detection-table td {
        padding: 12px 20px;
        border-top: 1px solid rgba(255,255,255,0.05);
        font-size: 1rem;
        color: #e0e0e0;
    }

    .detection-table tr:hover td {
        background: rgba(230, 57, 70, 0.08);
    }

    .color-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 12px;
        vertical-align: middle;
        box-shadow: 0 0 8px currentColor;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(14, 17, 23, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #444;
    }

    /* Animations */
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the YOLO model once and cache it."""
    from ultralytics import YOLO
    # Set logging to minimum to avoid cluttering logs
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    model = YOLO(str(MODEL_PATH))
    return model


# ── Drawing helpers ──────────────────────────────────────────────────────
def draw_detections(image: Image.Image, boxes, classes, confs, names,
                    show_labels: bool = True, show_conf: bool = True,
                    line_width: int = 2) -> Image.Image:
    """Draw bounding boxes on a PIL image and return the annotated copy."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to load a nicer font, fall back to default
    try:
        font_size = max(14, min(img.width, img.height) // 45)
        # Using a slightly bolder font if possible
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
        name = names[int(cls_id)]
        color = COLORS.get(name, (200, 200, 200))

        # Draw box with rounded-like appearance (multiple rectangles)
        for i in range(line_width):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # Label
        if show_labels:
            label = name
            if show_conf:
                label += f" {conf:.0%}"

            # Get text size
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 6

            # Label background (positioned smartly)
            label_y = y1 - th - pad * 2
            if label_y < 0:
                label_y = y2 + 2

            draw.rectangle(
                [x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2],
                fill=color
            )
            draw.text(
                (x1 + pad, label_y + pad),
                label, fill=(255, 255, 255), font=font
            )

    return img


def run_inference(model, image: Image.Image, conf: float, iou: float,
                  max_det: int, imgsz: int = 640) -> dict:
    """Run YOLO inference and return structured results."""
    # Convert PIL to numpy for ultralytics
    img_array = np.array(image)

    results = model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        device="cpu",
        save=False,
        verbose=False,
    )

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    names = r.names

    # Build counts
    counts = Counter(names[int(c)] for c in classes)

    return {
        "boxes": boxes,
        "classes": classes,
        "confs": confs,
        "names": names,
        "counts": counts,
        "total": len(boxes),
    }


def build_summary_html(counts: Counter, total: int) -> str:
    """Build an HTML summary table of detections."""
    hex_colors = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in COLORS.items()}

    rows = ""
    for name in CATEGORY_ORDER:
        count = counts.get(name, 0)
        if count == 0:
            continue
        pct = count / total * 100 if total > 0 else 0
        emoji = CATEGORY_EMOJI.get(name, "")
        color = hex_colors.get(name, "#ccc")
        rows += f"""
        <tr>
            <td>
                <span class="color-dot" style="background:{color}; color:{color};"></span>
                {emoji} {name}
            </td>
            <td style="text-align:center; font-weight:700;">{count}</td>
            <td style="text-align:right; color:#888; font-family: monospace;">{pct:.1f}%</td>
        </tr>
        """

    # WBC total
    wbc_total = sum(counts.get(w, 0) for w in WBC_SUBTYPES)
    if wbc_total > 0:
        wbc_pct = wbc_total / total * 100 if total > 0 else 0
        rows += f"""
        <tr style="border-top: 2px solid rgba(230, 57, 70, 0.2);">
            <td style="font-weight:700; color:#E63946;">🧬 Total WBC</td>
            <td style="text-align:center; font-weight:800; color:#E63946;">{wbc_total}</td>
            <td style="text-align:right; color:#888; font-family: monospace;">{wbc_pct:.1f}%</td>
        </tr>
        """

    html = f"""
    <table class="detection-table">
        <thead>
            <tr>
                <th>Cell Type</th>
                <th style="text-align:center;">Count</th>
                <th style="text-align:right;">Proportion</th>
            </tr>
        </thead>
        <tbody>
            {rows}
            <tr style="background: rgba(230, 57, 70, 0.05);">
                <td style="font-weight:800; border-bottom: none;">📊 Total Detections</td>
                <td style="text-align:center; font-weight:800; font-size:1.2rem; border-bottom: none;">{total}</td>
                <td style="text-align:right; color:#888; border-bottom: none;">100%</td>
            </tr>
        </tbody>
    </table>
    """
    return html


def build_wbc_differential_html(counts: Counter) -> str:
    """Build WBC differential count table (only WBC subtypes, normalized to 100%)."""
    hex_colors = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in COLORS.items()}
    wbc_total = sum(counts.get(w, 0) for w in WBC_SUBTYPES)
    if wbc_total == 0:
        return ""

    rows = ""
    for name in ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]:
        count = counts.get(name, 0)
        pct = count / wbc_total * 100 if wbc_total > 0 else 0
        color = hex_colors.get(name, "#ccc")
        emoji = CATEGORY_EMOJI.get(name, "")

        # Bar visualization
        rows += f"""
        <tr>
            <td>
                <span class="color-dot" style="background:{color}; color:{color};"></span>
                {emoji} {name}
            </td>
            <td style="text-align:center; font-weight:700;">{count}</td>
            <td style="text-align:center; font-weight:700;">{pct:.1f}%</td>
            <td>
                <div style="background:rgba(255,255,255,0.05); border-radius:10px; overflow:hidden; height:10px; width: 100px;">
                    <div style="background:{color}; width:{pct}%; height:100%; border-radius:10px;
                                box-shadow: 0 0 10px {color};"></div>
                </div>
            </td>
        </tr>
        """

    html = f"""
    <table class="detection-table">
        <thead>
            <tr>
                <th>WBC Subtype</th>
                <th style="text-align:center;">Count</th>
                <th style="text-align:center;">Differential</th>
                <th>Distribution</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    return html


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    with st.expander("🔬 Advanced Inference", expanded=True):
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum confidence score to keep a detection. Recommended: 0.20–0.35."
        )

        iou_threshold = st.slider(
            "IoU (NMS)",
            min_value=0.1,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression. Higher = more overlapping boxes allowed."
        )

        imgsz_val = st.select_slider(
            "Inference resolution",
            options=[320, 416, 512, 640, 800, 1024],
            value=640,
            help="Higher resolution improves detection of small cells (Platelets) but increases processing time."
        )

        max_det_val = st.slider(
            "Max detections",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
        )

    st.markdown("---")
    st.markdown("## 🎨 Display Options")
    
    with st.expander("Visual Settings", expanded=False):
        show_labels = st.checkbox("Show labels", value=True)
        show_conf = st.checkbox("Show confidence", value=True)
        line_width = st.slider("Box thickness", 1, 5, 2)

    st.markdown("---")
    st.markdown("## 📝 Report Metadata")
    
    with st.expander("Lab Info", expanded=False):
        patient_id = st.text_input("Patient ID / Reference", placeholder="e.g. PAT-2026-001")
        lab_notes = st.text_area("Analysis Notes", placeholder="Enter clinical observations...")

    st.markdown("---")
    if st.button("🔄 Reset App", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("## ℹ️ About Model")
    st.info(
        f"""
        **Architecture:** YOLO26-m  
        **Input Size:** {imgsz_val}x{imgsz_val} px  
        **mAP@50:** 0.875  
        """
    )
    st.warning("⚠️ **Not for clinical use.** For research and educational purposes only.")


# ── Main content ─────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Blood Cell Detector</h1>
    <p>Premium AI-powered solution for automated peripheral blood smear analysis.
    Leveraging advanced object detection to classify cells with laboratory-grade precision.</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🧠 Initializing Neural Engine..."):
    model = load_model()

# ── Image input ──────────────────────────────────────────────────────────
tab_upload, tab_sample, tab_info = st.tabs(["📤 Upload Image", "🖼️ Sample Images", "📚 How it Works"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a blood smear image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if not uploaded_file:
        st.info("💡 **Pro-tip:** Drag and drop high-resolution images for best results.")

with tab_sample:
    st.markdown("Select a sample blood smear image to explore the model's performance:")
    sample_images = sorted(TEST_IMAGES_DIR.glob("*.png")) + sorted(TEST_IMAGES_DIR.glob("*.jpg"))

    if sample_images:
        cols = st.columns(len(sample_images) if len(sample_images) < 4 else 4)
        for idx, img_path in enumerate(sample_images):
            with cols[idx % 4]:
                sample_img = Image.open(img_path)
                st.image(sample_img, caption=img_path.stem.replace("_", " ").title(), use_container_width=True)
                if st.button(f"🔍 Analyze Sample", key=f"sample_{idx}", use_container_width=True):
                    st.session_state["selected_sample"] = str(img_path)

with tab_info:
    st.markdown("""
    ### 🧬 The Science Behind the Detector
    
    This application uses a fine-tuned **YOLO26 (You Only Look Once)** object detection model. 
    The neural network has been trained on thousands of annotated blood smear images to recognize the 
    distinct morphological features of different cell types.
    
    #### 🔍 Detection Process
    1. **Preprocessing**: The image is resized to 640x640 pixels and normalized.
    2. **Feature Extraction**: The model's backbone identifies patterns like nuclear shape, cytoplasm color, and cell size.
    3. **Classification**: Each detected cell is assigned to one of 7 categories.
    4. **Post-processing**: Non-Maximum Suppression (NMS) removes duplicate detections.
    
    #### 🔴 Detected Categories
    - **RBC**: Erythrocytes (Red Blood Cells)
    - **Platelets**: Thrombocytes for clotting.
    - **WBC Subtypes**: The 5 standard types (Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil).
    """)
    st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=64)

# Determine which image to process
image_to_process = None
image_source = None

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    image_source = uploaded_file.name
elif "selected_sample" in st.session_state:
    sample_path = st.session_state["selected_sample"]
    image_to_process = Image.open(sample_path).convert("RGB")
    image_source = Path(sample_path).name

# ── Run detection ────────────────────────────────────────────────────────
if image_to_process is not None:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("---")

    # Run inference
    with st.spinner("⚡ Processing Neural Pipeline..."):
        results = run_inference(model, image_to_process, conf_threshold,
                                iou_threshold, max_det_val, imgsz_val)

    # Draw annotations
    annotated_img = draw_detections(
        image_to_process,
        results["boxes"],
        results["classes"],
        results["confs"],
        results["names"],
        show_labels=show_labels,
        show_conf=show_conf,
        line_width=line_width,
    )

    # Layout: image + results side by side
    col_img, col_results = st.columns([3, 2])

    with col_img:
        st.markdown(f"### 📸 Analysis Results")
        st.markdown(f"*Source: `{image_source}`*")

        # Toggle between original and annotated
        view_mode = st.segmented_control(
            "View Mode",
            ["Annotated", "Original"],
            default="Annotated",
            label_visibility="collapsed",
        )
        if view_mode == "Annotated":
            st.image(annotated_img, use_container_width=True)
        else:
            st.image(image_to_process, use_container_width=True)
            
        # Download button for the image
        import io
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='JPEG')
        st.download_button(
            label="📥 Download Annotated Image",
            data=img_byte_arr.getvalue(),
            file_name=f"detected_{image_source}",
            mime="image/jpeg",
            use_container_width=True
        )

    with col_results:
        st.markdown("### 📊 Detection Summary")

        # Stat cards row
        rbc_count = results["counts"].get("RBC", 0)
        plt_count = results["counts"].get("Platelets", 0)
        wbc_count = sum(results["counts"].get(w, 0) for w in WBC_SUBTYPES)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{rbc_count}</div>
                <div class="stat-label">🔴 RBC</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{wbc_count}</div>
                <div class="stat-label">🔵 WBC</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{plt_count}</div>
                <div class="stat-label">🟢 Platelets</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed counts table
        summary_html = build_summary_html(results["counts"], results["total"])
        st.markdown(summary_html, unsafe_allow_html=True)
        
        # Download Report CSV
        import pandas as pd
        report_data = [
            {"Cell Type": k, "Count": v, "Percentage": f"{(v/results['total']*100):.1f}%"}
            for k, v in results["counts"].items()
        ]
        # Add metadata to CSV if provided
        if patient_id or lab_notes:
            report_data.append({}) # empty row
            report_data.append({"Cell Type": "METADATA", "Count": "", "Percentage": ""})
            if patient_id:
                report_data.append({"Cell Type": "Patient ID", "Count": patient_id, "Percentage": ""})
            if lab_notes:
                report_data.append({"Cell Type": "Notes", "Count": lab_notes, "Percentage": ""})
                
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download CSV Report",
            data=csv,
            file_name=f"report_{image_source.split('.')[0]}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # WBC differential (full width below)
    wbc_total = sum(results["counts"].get(w, 0) for w in WBC_SUBTYPES)
    if wbc_total > 0:
        st.markdown("---")
        st.markdown("### 🧬 WBC Differential Count")
        st.markdown(
            "*Proportional distribution of leukocyte subtypes. "
            "Essential for clinical differential diagnosis.*"
        )
        diff_html = build_wbc_differential_html(results["counts"])
        st.markdown(diff_html, unsafe_allow_html=True)

    # Per-detection details (expandable)
    with st.expander("📋 Detailed Detection Log", expanded=False):
        if results["total"] > 0:
            det_data = []
            for i, (box, cls_id, conf) in enumerate(zip(
                    results["boxes"], results["classes"], results["confs"])):
                name = results["names"][int(cls_id)]
                x1, y1, x2, y2 = box
                det_data.append({
                    "Index": i + 1,
                    "Class": name,
                    "Confidence": f"{conf:.1%}",
                    "BBox (x1, y1, x2, y2)": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                })
            st.dataframe(det_data, use_container_width=True, hide_index=True)
        else:
            st.info("No cells detected with current settings.")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);">
        <div style="font-size: 5rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(230, 57, 70, 0.2));">🔬</div>
        <h2 style="color: #fff; font-weight: 700; margin-bottom: 1rem;">Ready for Analysis</h2>
        <p style="color: #888; max-width: 600px; margin: 0 auto; font-size: 1.1rem;">
            Upload a blood smear image or choose a sample to start the AI-powered detection.
            The system will automatically identify RBC, Platelets, and WBC subtypes.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>🔬 <strong>Blood Cell Detector v1.0</strong><br>
    Powered by YOLO26 Neural Engine · Optimized for CPU Inference<br>
    © 2026 Advanced Bioinformatic Solutions</p>
</div>
""", unsafe_allow_html=True)
