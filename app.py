import base64
import io
import time
from datetime import datetime
from html import escape

import streamlit as st
from PIL import Image

from huggingface.api_client import classify_image, MODELS, DEFAULT_MODEL
from ai_engine.classifier import get_classification_result

MAX_HISTORY = 12
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Styling adapted from the original Dash interface for Streamlit.
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
    :root { color-scheme: dark; }
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background:
          radial-gradient(ellipse at 15% 15%, rgba(80,70,220,.20), transparent 36%),
          radial-gradient(ellipse at 85% 25%, rgba(0,205,220,.12), transparent 32%),
          linear-gradient(135deg,#090b17 0%,#101326 52%,#0a1220 100%);
        color: #f4f6ff;
    }
    .block-container { max-width: 1400px; padding-top: 1.4rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    .brand { display:flex; align-items:center; gap:12px; font-size:1.05rem; font-weight:800; color:#f7f8ff; }
    .brand-logo {
        width:44px; height:44px; border-radius:14px; display:flex; align-items:center;
        justify-content:center; background:linear-gradient(135deg,#6d5dfc,#18c8e5);
        color:white; font-size:1.35rem; box-shadow:0 8px 28px rgba(72,111,255,.25);
    }
    .brand-sub { display:block; font-size:.72rem; font-weight:500; color:#9ba7c5; margin-top:2px; }
    .status-pill { display:inline-flex; align-items:center; gap:8px; padding:8px 12px;
        border:1px solid rgba(160,175,220,.20); border-radius:999px; color:#aeb9d8; font-size:.78rem; }
    .status-dot { width:7px; height:7px; background:#36d9a4; border-radius:50%; display:inline-block; }
    .hero { text-align:center; padding:2.5rem 0 1.5rem; }
    .eyebrow { color:#91a4ff; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }
    .hero-title { font-size:clamp(2.3rem,5vw,4.1rem); font-weight:800; letter-spacing:-.055em; line-height:1.08; margin:.8rem 0; color:#f7f8ff; }
    .gradient-text { background:linear-gradient(90deg,#8b7cff,#31d7e9); -webkit-background-clip:text; color:transparent; }
    .hero-sub { max-width:760px; margin:0 auto; color:#a9b4d0; font-size:1rem; line-height:1.7; }
    .stat-card { min-height:98px; border:1px solid rgba(166,180,225,.17); background:rgba(19,25,47,.72);
        border-radius:18px; padding:18px 20px; display:flex; gap:14px; align-items:center; margin-bottom:1.2rem; }
    .stat-icon { width:44px; height:44px; display:flex; align-items:center; justify-content:center;
        border-radius:14px; background:rgba(117,101,255,.16); color:#a79aff; font-size:1.2rem; }
    .stat-value { font-size:1.4rem; font-weight:800; color:#f5f6ff; overflow-wrap:anywhere; }
    .stat-label { font-size:.76rem; color:#9ba7c5; margin-top:2px; }
    .glass-card { border:1px solid rgba(166,180,225,.18); border-radius:22px;
        background:linear-gradient(145deg,rgba(23,29,53,.90),rgba(13,19,37,.86));
        padding:24px; box-shadow:0 18px 50px rgba(0,0,0,.14); min-height:100%; }
    .card-title { color:#f6f7ff; font-size:1.15rem; font-weight:800; margin:0; }
    .card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; gap:12px; }
    .step-badge { border:1px solid rgba(150,166,220,.2); border-radius:999px; padding:5px 10px; color:#98a6c9; font-size:.72rem; white-space:nowrap; }
    .section-label { font-size:.78rem; color:#a8b5d7; text-transform:uppercase; letter-spacing:.08em; font-weight:800; margin:24px 0 12px; }
    .result-hero { display:flex; align-items:center; gap:24px; padding:20px; border:1px solid rgba(80,220,200,.18);
        background:rgba(30,194,187,.055); border-radius:18px; }
    .gauge { width:100px; height:100px; flex:0 0 100px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        background:conic-gradient(#42d6d2 calc(var(--p)*1%), rgba(125,145,185,.16) 0); position:relative; }
    .gauge:before { content:''; position:absolute; inset:7px; border-radius:50%; background:#131a31; }
    .gauge-inner { position:relative; text-align:center; }
    .gauge-value { font-size:1.35rem; font-weight:800; color:#f5f7ff; }
    .gauge-label { font-size:.68rem; color:#a7b3d1; }
    .result-kicker { color:#52d7c2; font-size:.75rem; font-weight:700; margin-bottom:5px; }
    .result-label { font-size:1.6rem; font-weight:800; color:#fff; overflow-wrap:anywhere; }
    .result-aliases { color:#a4b0ce; font-size:.83rem; margin-top:3px; overflow-wrap:anywhere; }
    .confidence-tag { display:inline-block; margin-top:10px; padding:5px 9px; border-radius:999px; font-size:.73rem; }
    .confidence-tag.high { color:#65e0b6; background:rgba(54,217,164,.12); }
    .confidence-tag.medium { color:#f2c879; background:rgba(242,200,121,.12); }
    .confidence-tag.low { color:#b9c4e0; background:rgba(150,166,205,.12); }
    .bar-item { margin:14px 0; }
    .bar-head { display:flex; justify-content:space-between; gap:12px; margin-bottom:7px; color:#dce3f8; font-size:.84rem; }
    .bar-name { overflow-wrap:anywhere; }
    .bar-rank { color:#8796bd; margin-right:9px; font-size:.74rem; }
    .bar-pct { color:#aebbe0; white-space:nowrap; font-variant-numeric:tabular-nums; }
    .bar-track { height:8px; border-radius:999px; background:rgba(132,150,193,.15); overflow:hidden; }
    .bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#7867ff,#35d4e4); }
    .meta-chip { display:inline-flex; gap:7px; align-items:center; margin:8px 7px 0 0; border:1px solid rgba(155,172,220,.18);
        border-radius:999px; padding:6px 10px; color:#aeb9d5; font-size:.72rem; }
    .history-card { border:1px solid rgba(155,172,220,.17); border-radius:15px; padding:12px; background:rgba(18,25,46,.8); margin-bottom:10px; }
    .history-label { font-weight:700; color:#f0f3ff; overflow-wrap:anywhere; }
    .history-meta { display:flex; justify-content:space-between; gap:8px; color:#a3afce; font-size:.76rem; margin-top:5px; }
    .history-pct { color:#52d7c2; font-weight:800; }
    .site-footer { text-align:center; color:#8794b6; font-size:.8rem; padding:30px 0 5px; }
    .site-footer a { color:#9c91ff; }
    div[data-testid="stFileUploader"] { border:1px dashed rgba(142,160,220,.4); border-radius:16px; padding:10px; background:rgba(9,14,30,.38); }
    div[data-testid="stButton"] button { border-radius:12px; font-weight:700; min-height:42px; }
    div[data-testid="stSelectbox"] label, div[data-testid="stFileUploader"] label { color:#c3cde6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def split_label(label):
    """Convert 'tabby, tabby cat' to ('tabby', 'tabby cat')."""
    parts = [part.strip() for part in str(label).split(",") if part.strip()]
    if not parts:
        return str(label), ""
    primary = parts[0]
    aliases = [part for part in parts[1:] if part.lower() != primary.lower()]
    return primary, ", ".join(aliases)


def confidence_level(confidence):
    if confidence >= 70:
        return "high", "High confidence"
    if confidence >= 40:
        return "medium", "Moderate confidence"
    return "low", "Low confidence"


def make_thumbnail(image_bytes, size=320):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_results(result, model, elapsed):
    primary, aliases = split_label(result["label"])
    confidence = float(result["confidence"])
    level, level_text = confidence_level(confidence)

    st.markdown(
        f"""
        <div class="result-hero">
          <div class="gauge" style="--p:{max(0, min(100, confidence))}">
            <div class="gauge-inner">
              <div class="gauge-value">{confidence:.0f}%</div>
              <div class="gauge-label">Confidence</div>
            </div>
          </div>
          <div>
            <div class="result-kicker">✓ Top prediction</div>
            <div class="result-label">{escape(primary)}</div>
            <div class="result-aliases">{("Also known as: " + escape(aliases)) if aliases else ""}</div>
            <span class="confidence-tag {level}">{level_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">All predictions</div>', unsafe_allow_html=True)
    for index, item in enumerate(result.get("all_results", []), start=1):
        name, _ = split_label(item["label"])
        pct = max(0.0, min(100.0, float(item["confidence"])))
        st.markdown(
            f"""
            <div class="bar-item">
              <div class="bar-head">
                <span class="bar-name"><span class="bar-rank">#{index}</span>{escape(name)}</span>
                <span class="bar-pct">{pct:.2f}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{max(pct, 1.5):.2f}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<span class="meta-chip">⚙ {escape(str(MODELS.get(model, model)))}</span>'
        f'<span class="meta-chip">ϟ {elapsed:.2f}s</span>'
        f'<span class="meta-chip">◷ {datetime.now().strftime("%H:%M:%S")}</span>',
        unsafe_allow_html=True,
    )


if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

# Navigation / brand
nav_left, nav_right = st.columns([3, 1])
with nav_left:
    st.markdown(
        '<div class="brand"><div class="brand-logo">◉</div><div>AI Image Classifier'
        '<span class="brand-sub">Powered by Hugging Face</span></div></div>',
        unsafe_allow_html=True,
    )
with nav_right:
    st.markdown(
        '<div style="text-align:right;padding-top:10px"><span class="status-pill">'
        '<span class="status-dot"></span>Ready · Hugging Face</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hero"><div class="eyebrow">✦ Vision Transformers · Image inference</div>'
    '<div class="hero-title">See the world through <span class="gradient-text">AI eyes</span></div>'
    '<div class="hero-sub">Drop in any photo and state-of-the-art vision models will identify '
    'what’s in it, ranked by confidence, in seconds.</div></div>',
    unsafe_allow_html=True,
)

# Statistics
history = st.session_state.history
average_confidence = (
    sum(float(item["confidence"]) for item in history) / len(history)
    if history else None
)
stat1, stat2, stat3 = st.columns(3)
with stat1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-icon">▧</div><div>'
        f'<div class="stat-value">{len(history)}</div><div class="stat-label">Images analyzed</div></div></div>',
        unsafe_allow_html=True,
    )
with stat2:
    avg_text = f"{average_confidence:.1f}%" if average_confidence is not None else "--"
    st.markdown(
        f'<div class="stat-card"><div class="stat-icon">↗</div><div>'
        f'<div class="stat-value">{avg_text}</div><div class="stat-label">Avg confidence</div></div></div>',
        unsafe_allow_html=True,
    )
with stat3:
    active_model_name = MODELS.get(st.session_state.model, st.session_state.model)
    st.markdown(
        f'<div class="stat-card"><div class="stat-icon">⚙</div><div>'
        f'<div class="stat-value">{escape(str(active_model_name))}</div>'
        f'<div class="stat-label">Active model</div></div></div>',
        unsafe_allow_html=True,
    )

left_col, right_col = st.columns([5, 7], gap="large")

with left_col:
    st.markdown(
        '<div class="glass-card"><div class="card-head">'
        '<div class="card-title">⇧ Upload image</div><span class="step-badge">Step 1</span></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your image here or browse",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=False,
        help="PNG, JPG, WEBP, GIF · up to 10 MB",
    )

    upload_valid = False
    image_bytes = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            st.error("The image exceeds the 10 MB limit. Please choose a smaller image.")
        else:
            try:
                with Image.open(io.BytesIO(image_bytes)) as preview_image:
                    preview_image.load()
                    width, height = preview_image.size
                    st.image(preview_image, caption=uploaded_file.name, use_container_width=True)
                st.caption(
                    f"{uploaded_file.name} · {format_size(len(image_bytes))} · {width} × {height}"
                )
                upload_valid = True
            except Exception:
                st.error("This file could not be read as an image.")

    model_ids = list(MODELS.keys())
    if not model_ids:
        st.error("No models are configured in huggingface.api_client.")
        st.stop()
    default_index = model_ids.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_ids else 0
    selected_model = st.selectbox(
        "Model",
        options=model_ids,
        index=default_index,
        format_func=lambda model_id: MODELS.get(model_id, model_id),
    )
    st.session_state.model = selected_model

    button_col, clear_col = st.columns([4, 1])
    with button_col:
        classify_clicked = st.button(
            "✦  Classify image",
            type="primary",
            use_container_width=True,
            disabled=not upload_valid,
        )
    with clear_col:
        clear_clicked = st.button("✕", use_container_width=True, help="Clear image and result")

    if clear_clicked:
        st.session_state.result = None
        st.session_state.last_upload_id = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if classify_clicked and upload_valid and image_bytes is not None:
        try:
            with st.spinner("Analyzing image…"):
                started = time.perf_counter()
                raw_results = classify_image(image_bytes, model=selected_model)
                elapsed = time.perf_counter() - started
                classification = get_classification_result(raw_results)

            st.session_state.result = {
                "data": classification,
                "model": selected_model,
                "elapsed": elapsed,
                "filename": uploaded_file.name,
            }

            history_entry = {
                "thumb": make_thumbnail(image_bytes),
                "label": split_label(classification["label"])[0],
                "confidence": float(classification["confidence"]),
                "model": MODELS.get(selected_model, selected_model),
                "filename": uploaded_file.name,
                "time": datetime.now().strftime("%H:%M"),
            }
            st.session_state.history = (
                [history_entry] + st.session_state.history
            )[:MAX_HISTORY]
            st.rerun()
        except Exception as error:
            st.error(
                "We couldn't classify this image. Try again or pick another model. "
                f"Details: {error}"
            )

with right_col:
    st.markdown(
        '<div class="glass-card"><div class="card-head">'
        '<div class="card-title">◎ Results</div><span class="step-badge">Step 2</span></div>',
        unsafe_allow_html=True,
    )
    current_result = st.session_state.result
    if current_result is None:
        st.markdown(
            '<div style="text-align:center;padding:3.5rem 1rem;color:#a9b4d0">'
            '<div style="font-size:3rem;margin-bottom:12px">✧</div>'
            '<div style="font-size:1.2rem;font-weight:800;color:#f2f4ff">Waiting for an image</div>'
            '<div style="max-width:380px;margin:8px auto;line-height:1.7">'
            'Upload a photo and hit Classify. The AI will tell you what it sees and how confident it is.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        render_results(
            current_result["data"],
            current_result["model"],
            current_result["elapsed"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Recent history
st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
history_title_col, history_button_col = st.columns([4, 1])
with history_title_col:
    st.markdown('<div class="card-title">◷ Recent classifications</div>', unsafe_allow_html=True)
with history_button_col:
    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

history = st.session_state.history
if not history:
    st.info("No images classified yet. Your results will appear here.")
else:
    history_columns = st.columns(3)
    for index, item in enumerate(history):
        with history_columns[index % 3]:
            st.markdown('<div class="history-card">', unsafe_allow_html=True)
            st.image(item["thumb"], use_container_width=True)
            st.markdown(
                f'<div class="history-label">{escape(str(item["label"]))}</div>'
                f'<div class="history-meta"><span class="history-pct">{float(item["confidence"]):.1f}%</span>'
                f'<span>{escape(str(item["time"]))}</span></div>'
                f'<div class="history-meta"><span>{escape(str(item["model"]))}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="site-footer">Built with Streamlit · Models from '
    '<a href="https://huggingface.co" target="_blank">Hugging Face</a></div>',
    unsafe_allow_html=True,
)
