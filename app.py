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
