from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import base64
import io
import time
from datetime import datetime

from PIL import Image

from huggingface.api_client import classify_image, MODELS, DEFAULT_MODEL
from ai_engine.classifier import get_classification_result


MAX_HISTORY = 12


# =========================
# CREATE APP
# =========================

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"
    ]
)

app.title = "AI Image Classifier"

# Apply the saved theme before first paint to avoid a flash of the wrong theme
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            try {
                var t = JSON.parse(localStorage.getItem("theme-store"));
                if (t) document.documentElement.dataset.theme = t;
            } catch (e) {}
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


# =========================
# HELPERS
# =========================

def split_label(label):
    """'tabby, tabby cat' -> ('tabby', 'tabby cat')"""

    parts = [p.strip() for p in label.split(",") if p.strip()]

    if not parts:
        return label, ""

    primary = parts[0]
    aliases = [p for p in parts[1:] if p.lower() != primary.lower()]

    return primary, ", ".join(aliases)


def confidence_level(confidence):

    if confidence >= 70:
        return "high", "High confidence", "bi-check-circle-fill"

    if confidence >= 40:
        return "medium", "Moderate confidence", "bi-exclamation-circle-fill"

    return "low", "Low confidence", "bi-question-circle-fill"


def make_thumbnail(image_bytes, size=320):

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image.thumbnail((size, size))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def format_size(num_bytes):

    for unit in ["B", "KB", "MB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024

    return f"{num_bytes:.1f} GB"


def meta_chip(icon, text):

    return html.Span(
        [html.I(className=f"bi {icon}"), text],
        className="meta-chip",
        title=text
    )


# =========================
# UI BUILDING BLOCKS
# =========================

def dropzone_empty():

    return html.Div(
        [
            html.Div(html.I(className="bi bi-cloud-arrow-up-fill"), className="drop-icon"),
            html.Div(["Drop your image here or ", html.Span("browse")], className="drop-title"),
            html.Div("PNG, JPG, WEBP, GIF · up to 10 MB", className="drop-hint")
        ],
        className="dropzone-empty"
    )


def dropzone_preview(contents):

    return html.Div(
        [
            html.Img(src=contents, className="preview-img"),
            html.Div(
                [html.I(className="bi bi-arrow-repeat me-1"), "Click or drop to replace"],
                className="preview-overlay"
            )
        ],
        className="preview-wrap"
    )


def empty_state():

    return html.Div(
        [
            html.Div(
                [
                    html.Div(className="empty-ring"),
                    html.Div(className="empty-ring r2"),
                    html.Div(html.I(className="bi bi-stars"), className="empty-core")
                ],
                className="empty-visual"
            ),
            html.Div("Waiting for an image", className="empty-title"),
            html.Div(
                "Upload a photo and hit Classify. The AI will tell you what it sees "
                "and how confident it is.",
                className="empty-text"
            )
        ],
        className="empty-state"
    )


def error_state(message):

    return html.Div(
        [
            html.Div(html.I(className="bi bi-exclamation-triangle-fill"), className="error-icon"),
            html.Div("Something went wrong", className="empty-title"),
            html.Div("We couldn't classify this image. Try again or pick another model.",
                     className="empty-text"),
            html.Div(message, className="error-detail")
        ],
        className="error-state"
    )


def results_view(result, model, elapsed):

    primary, aliases = split_label(result["label"])
    confidence = result["confidence"]
    level, level_text, level_icon = confidence_level(confidence)

    bars = []

    for index, item in enumerate(result["all_results"]):

        name, _ = split_label(item["label"])

        bars.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                [html.Span(f"#{index + 1}", className="bar-rank"), name],
                                className="bar-name",
                                title=item["label"]
                            ),
                            html.Span(f"{item['confidence']:.2f}%", className="bar-pct")
                        ],
                        className="bar-head"
                    ),
                    html.Div(
                        html.Div(
                            className="bar-fill",
                            style={"--w": f"{max(item['confidence'], 1.5)}%"}
                        ),
                        className="bar-track"
                    )
                ],
                className="bar-item top" if index == 0 else "bar-item",
                style={"animationDelay": f"{0.15 + index * 0.1}s"}
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Div(
                            [
                                html.Div(f"{confidence:.0f}%", className="gauge-value"),
                                html.Div("Confidence", className="gauge-label")
                            ],
                            className="gauge-inner"
                        ),
                        className="gauge",
                        style={"--p": confidence}
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.I(className="bi bi-patch-check-fill"), "Top prediction"],
                                className="result-kicker"
                            ),
                            html.Div(primary, className="result-label"),
                            html.Div(
                                f"Also known as: {aliases}" if aliases else "",
                                className="result-aliases"
                            ),
                            html.Span(
                                [html.I(className=f"bi {level_icon}"), level_text],
                                className=f"confidence-tag {level}"
                            )
                        ],
                        className="result-info"
                    )
                ],
                className="result-hero"
            ),

            html.Div("All predictions", className="section-label"),
            html.Div(bars),

            html.Div(
                [
                    meta_chip("bi-cpu", MODELS.get(model, model)),
                    meta_chip("bi-lightning-charge-fill", f"{elapsed:.2f}s"),
                    meta_chip("bi-clock", datetime.now().strftime("%H:%M:%S"))
                ],
                className="result-footer"
            )
        ],
        style={"display": "flex", "flexDirection": "column", "flex": 1}
    )


def stat_card(icon, value_id, label, value, variant=""):

    return html.Div(
        [
            html.Div(html.I(className=f"bi {icon}"), className=f"stat-icon {variant}"),
            html.Div(
                [
                    html.Div(value, id=value_id, className="stat-value"),
                    html.Div(label, className="stat-label")
                ],
                style={"minWidth": 0}
            )
        ],
        className="stat-card"
    )


scan_loader = html.Div(
    [
        html.Div(html.I(className="bi bi-image"), className="scan-box"),
        html.Div("Analyzing image", className="scan-text")
    ],
    className="scan-loader"
)


# =========================
# LAYOUT
# =========================

app.layout = html.Div(
    [
        dcc.Store(id="theme-store", storage_type="local", data="dark"),
        dcc.Store(id="history-store", storage_type="session", data=[]),

        # ANIMATED BACKGROUND
        html.Div(
            [
                html.Div(className="bg-grid"),
                html.Div(className="orb orb-1"),
                html.Div(className="orb orb-2"),
                html.Div(className="orb orb-3")
            ],
            className="bg-layer"
        ),

        html.Div(
            [
                # NAVBAR
                html.Div(
                    dbc.Container(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(html.I(className="bi bi-eye-fill"), className="brand-logo"),
                                        html.Div(
                                            [
                                                "AI Image Classifier",
                                                html.Span("Powered by Hugging Face", className="brand-sub")
                                            ]
                                        )
                                    ],
                                    className="brand"
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            [html.Span(className="status-dot"), "Running locally · ONNX"],
                                            className="status-pill"
                                        ),
                                        html.Button(
                                            [
                                                html.I(className="bi bi-moon-stars-fill"),
                                                html.I(className="bi bi-sun-fill")
                                            ],
                                            id="theme-toggle",
                                            className="theme-toggle",
                                            title="Toggle theme"
                                        )
                                    ],
                                    className="nav-actions"
                                )
                            ],
                            className="nav-inner"
                        )
                    ),
                    className="top-nav"
                ),

                dbc.Container(
                    [
                        # HERO
                        html.Div(
                            [
                                html.Div(
                                    [html.I(className="bi bi-stars"), "Vision Transformers · On-device inference"],
                                    className="eyebrow"
                                ),
                                html.H1(
                                    ["See the world through ", html.Span("AI eyes", className="gradient-text")],
                                    className="hero-title"
                                ),
                                html.P(
                                    "Drop in any photo and state-of-the-art vision models will identify "
                                    "what's in it, ranked by confidence, in seconds.",
                                    className="hero-sub"
                                )
                            ],
                            className="hero"
                        ),

                        # STATS
                        html.Div(
                            [
                                stat_card("bi-images", "stat-count", "Images analyzed", "0"),
                                stat_card("bi-graph-up-arrow", "stat-avg", "Avg confidence", "--", "cyan"),
                                stat_card("bi-cpu", "stat-model", "Active model", MODELS[DEFAULT_MODEL], "pink")
                            ],
                            className="stats-row"
                        ),

                        # MAIN AREA
                        dbc.Row(
                            [
                                # LEFT: UPLOAD
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3(
                                                        [html.I(className="bi bi-upload"), "Upload image"],
                                                        className="card-title"
                                                    ),
                                                    html.Span("Step 1", className="step-badge")
                                                ],
                                                className="card-head"
                                            ),

                                            dcc.Upload(
                                                id="upload-image",
                                                children=dropzone_empty(),
                                                className="dropzone",
                                                accept="image/*",
                                                max_size=10 * 1024 * 1024,
                                                multiple=False
                                            ),

                                            html.Div(id="file-meta", className="file-meta"),

                                            html.Label("Model", className="field-label", htmlFor="model-select"),
                                            dbc.Select(
                                                id="model-select",
                                                options=[{"label": name, "value": model_id}
                                                         for model_id, name in MODELS.items()],
                                                value=DEFAULT_MODEL,
                                                className="model-select"
                                            ),

                                            html.Div(
                                                [
                                                    html.Button(
                                                        [html.I(className="bi bi-magic me-2"), "Classify image"],
                                                        id="classify-button",
                                                        className="btn-primary-glow",
                                                        disabled=True
                                                    ),
                                                    html.Button(
                                                        html.I(className="bi bi-x-lg"),
                                                        id="clear-button",
                                                        className="btn-ghost",
                                                        title="Clear image"
                                                    )
                                                ],
                                                className="btn-row"
                                            )
                                        ],
                                        className="glass-card"
                                    ),
                                    lg=5,
                                    className="mb-4"
                                ),

                                # RIGHT: RESULTS
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3(
                                                        [html.I(className="bi bi-bullseye"), "Results"],
                                                        className="card-title"
                                                    ),
                                                    html.Span("Step 2", className="step-badge")
                                                ],
                                                className="card-head"
                                            ),

                                            dcc.Loading(
                                                html.Div(empty_state(), id="results", className="results-body"),
                                                custom_spinner=scan_loader,
                                                delay_show=150
                                            )
                                        ],
                                        className="glass-card"
                                    ),
                                    lg=7,
                                    className="mb-4"
                                )
                            ]
                        ),

                        # HISTORY
                        html.Div(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3(
                                                [html.I(className="bi bi-clock-history"), "Recent classifications"],
                                                className="card-title"
                                            ),
                                            html.Button(
                                                [html.I(className="bi bi-trash3 me-1"), "Clear"],
                                                id="clear-history",
                                                className="link-btn"
                                            )
                                        ],
                                        className="card-head"
                                    ),
                                    html.Div(id="history-grid")
                                ],
                                className="glass-card"
                            ),
                            className="history-section"
                        ),

                        html.Footer(
                            [
                                "Built with Dash · Models from ",
                                html.A("Hugging Face", href="https://huggingface.co", target="_blank")
                            ],
                            className="site-footer"
                        )
                    ]
                )
            ],
            className="app-shell"
        )
    ]
)


# =========================
# THEME (client-side)
# =========================

app.clientside_callback(
    """
    function(n, theme) {
        return theme === "light" ? "dark" : "light";
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(theme) {
        document.documentElement.dataset.theme = theme || "dark";
        return theme === "light" ? "Switch to dark mode" : "Switch to light mode";
    }
    """,
    Output("theme-toggle", "title"),
    Input("theme-store", "data")
)


# =========================
# UPLOAD PREVIEW
# =========================

@app.callback(
    Output("upload-image", "children"),
    Output("file-meta", "children"),
    Output("classify-button", "disabled"),
    Output("results", "children", allow_duplicate=True),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True
)
def preview_upload(contents, filename):

    if contents is None:
        return dropzone_empty(), [], True, empty_state()

    image_bytes = base64.b64decode(contents.split(",", 1)[1])

    chips = [
        meta_chip("bi-file-earmark-image", filename or "image"),
        meta_chip("bi-hdd", format_size(len(image_bytes)))
    ]

    try:
        width, height = Image.open(io.BytesIO(image_bytes)).size
        chips.append(meta_chip("bi-aspect-ratio", f"{width} × {height}"))
    except Exception:
        pass

    return dropzone_preview(contents), chips, False, empty_state()


@app.callback(
    Output("upload-image", "contents"),
    Input("clear-button", "n_clicks"),
    prevent_initial_call=True
)
def clear_upload(n_clicks):

    return None


# =========================
# CLASSIFY
# =========================

@app.callback(
    Output("results", "children"),
    Output("history-store", "data"),
    Input("classify-button", "n_clicks"),
    State("upload-image", "contents"),
    State("upload-image", "filename"),
    State("model-select", "value"),
    State("history-store", "data"),
    prevent_initial_call=True
)
def classify_uploaded_image(n_clicks, contents, filename, model, history):

    if contents is None:
        return empty_state(), no_update

    try:

        image_bytes = base64.b64decode(contents.split(",", 1)[1])

        started = time.perf_counter()
        raw_results = classify_image(image_bytes, model=model)
        elapsed = time.perf_counter() - started

        result = get_classification_result(raw_results)

        entry = {
            "thumb": make_thumbnail(image_bytes),
            "label": split_label(result["label"])[0],
            "confidence": result["confidence"],
            "model": MODELS.get(model, model),
            "filename": filename,
            "time": datetime.now().strftime("%H:%M")
        }

        history = ([entry] + (history or []))[:MAX_HISTORY]

        return results_view(result, model, elapsed), history

    except Exception as error:

        return error_state(str(error)), no_update


# =========================
# HISTORY + STATS
# =========================

@app.callback(
    Output("history-store", "data", allow_duplicate=True),
    Input("clear-history", "n_clicks"),
    prevent_initial_call=True
)
def clear_history(n_clicks):

    return []


@app.callback(
    Output("history-grid", "children"),
    Output("stat-count", "children"),
    Output("stat-avg", "children"),
    Input("history-store", "data")
)
def render_history(history):

    history = history or []

    if not history:
        return (
            html.Div("No images classified yet. Your results will appear here.", className="history-empty"),
            "0",
            "--"
        )

    cards = [
        html.Div(
            [
                html.Img(src=item["thumb"], className="history-thumb", alt=item["label"]),
                html.Div(
                    [
                        html.Div(item["label"], className="history-label", title=item.get("filename")),
                        html.Div(
                            [
                                html.Span(f"{item['confidence']:.1f}%", className="history-pct"),
                                html.Span(item["time"])
                            ],
                            className="history-meta"
                        )
                    ],
                    className="history-body"
                )
            ],
            className="history-card",
            title=item["model"]
        )
        for item in history
    ]

    average = sum(item["confidence"] for item in history) / len(history)

    return html.Div(cards, className="history-grid"), str(len(history)), f"{average:.1f}%"


@app.callback(
    Output("stat-model", "children"),
    Input("model-select", "value")
)
def show_model(model):

    return MODELS.get(model, model)


# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=8052
    )
