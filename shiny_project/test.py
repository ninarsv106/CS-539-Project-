import os
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui
import json

# --------------------------------------------------------------------------- #
# Palette / theming
# --------------------------------------------------------------------------- #
INK, SURFACE, MUTED, ACCENT, WARM = "#0f1b24", "#ffffff", "#6b7b86", "#0d7c8a", "#c77a1a"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter+Tight:wght@400;500;700&display=swap');

html, body {{
    font-family: 'Inter Tight', system-ui, sans-serif;
    color: {INK};
    background-color: #f2f8fc !important;
}}

/* Main content area beneath the navigation bar */
.bslib-page-fill,
.bslib-page-navbar,
.container-fluid,
.tab-content,
.tab-pane,
.html-fill-container,
main {{
    background-color: #f2f8fc !important;
}}

/* Keep cards white so they remain visually distinct */
.card ,
.bslib-card {{
    --bs-card-bg: #ffffff;
    background-color: #ffffff !important;
    border: 1px solid #d8e3ea !important;
    box-shadow: 0 3px 10px rgba(15, 27, 36, 0.08);
}}

.card-header,
.card-body,
.card-footer {{
    background-color: #ffffff !important;
}}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: {ACCENT};
}}

h1, .card-header {{ font-weight: 700; letter-spacing: -.01em; }}
.lede {{ font-size: 1.05rem; color: {MUTED}; max-width: 62ch; }}
.block-img {{ width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 4/3; }}
.placeholder {{ display: grid; place-items: center; background: #eef1f3;
                color: {MUTED}; font-family: 'IBM Plex Mono', monospace; font-size: .8rem; }}
.hitcount {{ font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: {MUTED}; }}
.stars {{ color: {WARM}; letter-spacing: .08em; margin-right: .5rem; }}
.chip {{ display: inline-block; padding: .12rem .55rem; margin: .15rem .25rem 0 0;
         border: 1px solid #dfe4e7; border-radius: 999px; font-size: .75rem; color: {MUTED}; }}
.content-copy {{ max-width: 1050px; line-height: 1.65; }}
.section-card {{ margin-bottom: 1rem; }}
.section-card p:last-child {{ margin-bottom: 0; }}
.data-note {{ font-weight: 700; margin-top: 1.25rem; margin-bottom: .75rem; }}

/* Allow the business dropdown to extend over the dashboard cards. */
.dashboard-filters,
.dashboard-filters .card-body,
.dashboard-filters .row,
.dashboard-filters [class*="col-"] {{
    overflow: visible !important;
}}

.dashboard-filters {{
    position: relative;
    z-index: 100;
}}

.business-filter-column {{
    position: relative;
    z-index: 103;
}}

.task-filter-column {{
    position: relative;
    z-index: 104;
}}

/* Display approximately 10 business names before scrolling. */
#business_name-selectized + .selectize-dropdown,
#business_name + .selectize-control .selectize-dropdown {{
    z-index: 10000 !important;
}}

#business_name-selectized + .selectize-dropdown .selectize-dropdown-content,
#business_name + .selectize-control .selectize-dropdown-content,
.selectize-control.single .selectize-dropdown-content {{
    max-height: 420px !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
}}

.selectize-control.dropdown-active {{
    z-index: 10000 !important;
}}
"""

# --------------------------------------------------------------------------- #
# Home tab data: precomputed review-level and business-level emotion outputs
# --------------------------------------------------------------------------- #
REPO_URL = "https://github.com/ninarsv106/CS-539-Project-"

DATA_DIR = Path(__file__).resolve().parent.parent / "datafiles"

CLEAN_REVIEWS_PATH = DATA_DIR / "yelp_reviews_clean_CA.csv"
EMOTION_REVIEWS_PATH = DATA_DIR / "yelp_reviews_with_hf_emotions.csv"
EMOTION_SUMMARY_PATH = DATA_DIR / "yelp_all_business_emotion_summary.csv"
SENTIMENT_PREDICTION_PATH = DATA_DIR / "flan_t5" / "flan_t5_sentiment.csv"
SUMMARY_PATH = DATA_DIR / "flan_t5" / "flan_t5_summary.csv"
FLAN_T5_METRICS = DATA_DIR / "flan_t5"/ "metrics.json"

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

def flan_data_loader():
    try:
        sentiment = pd.read_csv(SENTIMENT_PREDICTION_PATH)
        summary = pd.read_csv(SUMMARY_PATH)
        sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
        sentiment = sentiment.dropna(subset=["date"])
    except Exception as e:
        raise e
    return sentiment, summary

def _safe_freq(freq):
    """pandas >= 2.2 wants 'ME'/'QE'/'YE'; older versions want 'M'/'Q'/'Y'.
    Try the modern alias, fall back to the legacy one so the app runs on both."""
    try:
        pd.Grouper(freq=freq)
        return freq
    except ValueError:
        return {"ME": "M", "QE": "Q", "YE": "Y"}.get(freq, freq)

FLAN_SENTIMENT, FLAN_SUMMARY = flan_data_loader()
SENTIMENT_BUSINESSES = sorted(FLAN_SENTIMENT["business_name"].dropna().unique().tolist())
SUMMARY_BUSINESSES = sorted(FLAN_SUMMARY["business_name"].dropna().unique().tolist())
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}

def require_data_file(path: Path, description: str) -> Path:
    """Raise a clear startup error when a required project data file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}\n\n"
            "Place the file in the repository's datafiles directory. "
            "These files contain precomputed model outputs; the Shiny app "
            "does not run the emotion model in real time."
        )
    return path


require_data_file(CLEAN_REVIEWS_PATH, "Cleaned Yelp review CSV")
require_data_file(EMOTION_REVIEWS_PATH, "Review-level emotion CSV")
require_data_file(EMOTION_SUMMARY_PATH, "Business-level emotion summary CSV")

# Used by the Data tab.
LOCAL_PATH_DATA = pl.read_csv(CLEAN_REVIEWS_PATH)

# The business-level file is small enough to load once at application startup.
EMOTION_SUMMARY = (
    pl.read_csv(EMOTION_SUMMARY_PATH)
    .with_columns(
        pl.col("business_name").cast(pl.Utf8).str.strip_chars(),
        pl.col("matching_business_id_count").cast(pl.Int64, strict=False),
        pl.col("review_count").cast(pl.Int64, strict=False),
        *[
            pl.col(emotion).cast(pl.Float64, strict=False)
            for emotion in EMOTIONS
        ],
    )
    .drop_nulls(subset=["business_name"])
    .filter(pl.col("business_name").str.len_chars() > 0)
    .unique(subset=["business_name"], keep="first")
    .sort("business_name")
)

BUSINESS_CHOICES = EMOTION_SUMMARY.get_column("business_name").to_list()

if not BUSINESS_CHOICES:
    raise ValueError(
        f"No business names were found in {EMOTION_SUMMARY_PATH}."
    )

DEFAULT_BUSINESS = (
    "Iron Horse Auto Body"
    if "Iron Horse Auto Body" in BUSINESS_CHOICES
    else BUSINESS_CHOICES[0]
)

# Keep the large review-level emotion file lazy. It is scanned only when the
# selected business changes, and the resulting filtered frame is cached by
# Shiny's reactive calculation for all seven histogram outputs.
EMOTION_REVIEWS_LAZY = pl.scan_csv(
    EMOTION_REVIEWS_PATH,
    infer_schema_length=20_000,
    ignore_errors=True,
).select(
    [
        pl.col("business_name").cast(pl.Utf8).str.strip_chars(),
        *[
            pl.col(emotion).cast(pl.Float64, strict=False)
            for emotion in EMOTIONS
        ],
    ]
)

WWW = Path(__file__).parent / "www"
WWW.mkdir(exist_ok=True)


def plot_average_emotion_distribution(
    summary_row: dict,
    business_name: str,
):
    """Plot the precomputed mean emotion probability for one business."""
    values = pd.Series(
        {
            emotion: float(summary_row[emotion])
            for emotion in EMOTIONS
        }
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(values.index, values.values, color=ACCENT)
    ax.set_title(
        f"Average Emotion Distribution for {business_name}",
        loc="left",
        fontweight="bold",
        color=INK,
    )
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Average Probability")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def plot_emotion_histogram(
    emotion_frame: pl.DataFrame,
    emotion: str,
    business_name: str,
):
    """Plot the review-level probability distribution for one emotion."""
    values = (
        emotion_frame
        .get_column(emotion)
        .drop_nulls()
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=50, color=ACCENT)
    ax.set_title(
        f"Distribution of {emotion.capitalize()} Scores for {business_name}",
        loc="left",
        fontweight="bold",
        color=INK,
    )
    ax.set_xlabel(f"{emotion.capitalize()} Probability")
    ax.set_ylabel("Number of Reviews")
    ax.set_xlim(0, 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Model results
# --------------------------------------------------------------------------- #
def _fake_results(seed: int, skill: float) -> dict:
    """Temporary demonstration results retained from the original app."""
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.binomial(1, 0.3, n)
    score = np.clip(rng.normal(0.5 + skill * (y - 0.5), 0.18), 0, 1)
    epochs = np.arange(1, 26)
    return {
        "kind": "binary_demo",
        "y_true": y,
        "y_score": score,
        "train_loss": 0.7 * np.exp(-epochs / 7) + 0.08,
        "val_loss": 0.7 * np.exp(-epochs / 9) + 0.12 + rng.normal(0, 0.008, 25),
        "epochs": epochs,
    }


DISTILBERT_CLASS_METRICS = pd.DataFrame(
    [
        {"Class": "Negative", "Precision": 0.864108, "Recall": 0.874963,
         "F1": 0.869502, "Support": 3423},
        {"Class": "Neutral", "Precision": 0.524565, "Recall": 0.464525,
         "F1": 0.492723, "Support": 1494},
        {"Class": "Positive", "Precision": 0.956021, "Recall": 0.965375,
         "F1": 0.960675, "Support": 13083},
        {"Class": "Macro average", "Precision": 0.781565, "Recall": 0.768288,
         "F1": 0.774300, "Support": 18000},
        {"Class": "Weighted average", "Precision": 0.902732, "Recall": 0.906611,
         "F1": 0.904497, "Support": 18000},
    ]
)

DISTILBERT_RESULTS = {
    "kind": "distilbert",
    "model_name": "distilbert-base-uncased",
    "accuracy": 0.9066111111111111,
    "precision_macro": 0.781565120448617,
    "recall_macro": 0.7682877206885262,
    "f1_macro": 0.7743000995906857,
    "precision_weighted": 0.9027318479934742,
    "recall_weighted": 0.9066111111111111,
    "f1_weighted": 0.9044972373422764,
    "test_loss": 0.2769743502140045,
    "confusion_matrix": np.array(
        [
            [2995, 303, 125],
            [344, 694, 456],
            [127, 326, 12630],
        ]
    ),
    "class_names": ["Negative", "Neutral", "Positive"],
    "class_metrics": DISTILBERT_CLASS_METRICS,
    "loss_image": WWW / "distilbert_yelp_ca_training_validation_loss.png",
    "confusion_image": WWW / "distilbert_yelp_ca_confusion_matrix.png",
}
FLANT5_CLASS_METRICS = pd.DataFrame(
    [
        {"Class": "Negative", "Precision": 0.9032258064516129, "Recall": 0.9081081081081082,
         "F1": 0.9056603773584906, "Support": 185.0},
        {"Class": "Neutral", "Precision": 0.632183908045977, "Recall":  0.5913978494623656,
         "F1": 0.6111111111111112, "Support": 93.0},
        {"Class": "Positive", "Precision":0.9656593406593407, "Recall": 0.9723374827109267,
         "F1": 0.968986905582357, "Support": 723.0},
        {"Class": "Macro average", "Precision":  0.8336896850523102, "Recall": 0.8239478134271335,
         "F1": 0.828586131350653, "Support":  1001.0},
        {"Class": "Weighted average", "Precision": 0.9231384424960315, "Recall": 0.9250749250749251,
         "F1": 0.9240340018788193, "Support": 1001.0},
    ]
)
FLANT5BASE_RESULTS = {
    "kind": "flan-t5",
    "model_name": "flan-t5-base",
    "accuracy": 0.9250749250749251,
    "precision_macro": 0.8336896850523102,
    "recall_macro": 0.8239478134271335,
    "f1_macro": 0.828586131350653,
    "precision_weighted": 0.9231384424960315,
    "recall_weighted": 0.9250749250749251,
    "f1_weighted": 0.9240340018788193,
    "test_loss": 0.6769743502140045,
    "confusion_matrix": {
        "labels": [
            "negative",
            "neutral",
            "positive"
        ],
        "matrix": [
            [
                168,
                14,
                3
            ],
            [
                16,
                55,
                22
            ],
            [
                2,
                18,
                703
            ]
        ]
    },
    "class_names": ["Negative", "Neutral", "Positive"],
    "class_metrics": DISTILBERT_CLASS_METRICS,
    "loss_image": WWW / "class_metric_report.png",
    "confusion_image": WWW / "flan_t5_confusion_matrix.png",
}

RESULTS = {
    "DistilBERT": DISTILBERT_RESULTS,
    "FlanT5": FLANT5BASE_RESULTS,
}


# --------------------------------------------------------------------------- #
# Metrics and plot helpers
# --------------------------------------------------------------------------- #
def roc_points(y_true, y_score):
    order = np.argsort(-y_score)
    y = y_true[order]
    tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - y) / max((1 - y).sum(), 1)]
    return fpr, tpr


def auc(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def confusion(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    return np.array(
        [
            [np.sum((y_true == true) & (pred == predicted)) for predicted in (0, 1)]
            for true in (0, 1)
        ]
    )


def binary_accuracy(res):
    cm = confusion(res["y_true"], res["y_score"])
    return float(np.trace(cm) / cm.sum())


def _frame(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dfe4e7")


def plot_static_notebook_image(path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    image = plt.imread(path)
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def plot_roc(res, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fpr, tpr = roc_points(res["y_true"], res["y_score"])
    ax.plot(fpr, tpr, color=ACCENT, lw=2.2)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#c3ccd1")
    if annotate:
        ax.annotate(
            f"AUC {auc(fpr, tpr):.3f}",
            (0.62, 0.22),
            fontsize=13,
            fontweight="bold",
            color=INK,
        )
    _frame(ax, "ROC curve", "False positive rate", "True positive rate")
    return fig


def plot_confusion(res, annotate=True):
    if res["kind"] == "distilbert" or res["kind"] == "flan-t5":
        image_path = res["confusion_image"]
        if image_path.exists():
            return plot_static_notebook_image(image_path)

        cm = res["confusion_matrix"]
        class_names = res["class_names"]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(class_names)), class_names)
        ax.set_yticks(np.arange(len(class_names)), class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(
            "DistilBERT Sentiment Classification Confusion Matrix",
            fontweight="bold",
        )
        if annotate:
            threshold = cm.max() * 0.55
            for (row, column), value in np.ndenumerate(cm):
                ax.text(
                    column,
                    row,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else INK,
                    fontweight="bold",
                )
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cm = confusion(res["y_true"], res["y_score"])
    ax.imshow(cm / cm.sum(), cmap="BuGn", vmin=0)
    ax.set_xticks([0, 1], ["Predicted negative", "Predicted positive"])
    ax.set_yticks([0, 1], ["True negative", "True positive"])
    if annotate:
        for (row, column), value in np.ndenumerate(cm):
            ax.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=SURFACE if value > cm.max() * 0.6 else INK,
            )
    _frame(ax, "Confusion matrix at threshold 0.50", "", "")
    ax.tick_params(labelsize=9)
    return fig


def plot_learning_curve(res, annotate=True):
    if res["kind"] == "distilbert":
        image_path = res["loss_image"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"DistilBERT loss plot not found: {image_path}. "
                "Place the supplied PNG inside shiny_project/www."
            )
        return plot_static_notebook_image(image_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["epochs"], res["train_loss"], color=ACCENT, lw=2, label="Training")
    ax.plot(res["epochs"], res["val_loss"], color=WARM, lw=2, label="Validation")
    if annotate:
        best = int(np.argmin(res["val_loss"]))
        ax.axvline(res["epochs"][best], color="#c3ccd1", ls="--", lw=1)
        ax.text(
            res["epochs"][best] + 0.4,
            res["val_loss"].max(),
            f"Best epoch {res['epochs'][best]}",
            fontsize=9,
            color=MUTED,
        )
    ax.legend(frameon=False, fontsize=9)
    _frame(ax, "Training and validation loss", "Epoch", "Log loss")
    return fig


def plot_class_metrics(res, annotate=True):
    if res["kind"] != "distilbert" and res["kind"] != "flan-t5":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(
            0.5,
            0.5,
            "Class-specific precision, recall, and F1\n"
            "are available for the DistilBERT model, and FLAN-T5 model.",
            ha="center",
            va="center",
            fontsize=13,
            color=MUTED,
        )
        ax.axis("off")
        return fig

    metrics = res["class_metrics"].iloc[:3]
    positions = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 4.8))
    precision_bars = ax.bar(
        positions - width,
        metrics["Precision"],
        width,
        label="Precision",
    )
    recall_bars = ax.bar(
        positions,
        metrics["Recall"],
        width,
        label="Recall",
    )
    f1_bars = ax.bar(
        positions + width,
        metrics["F1"],
        width,
        label="F1",
    )

    ax.set_xticks(positions, metrics["Class"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(
        "DistilBERT Class-Specific Precision, Recall, and F1",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3)

    if annotate:
        for bars in (precision_bars, recall_bars, f1_bars):
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


PLOTTERS = {
    "ROC curve": plot_roc,
    "Confusion matrix": plot_confusion,
    "Training and validation loss": plot_learning_curve,
    "Class precision, recall, and F1": plot_class_metrics,
}

MODEL_CHARTS = {
    "DistilBERT": [
        "Training and validation loss",
        "Confusion matrix",
        "Class precision, recall, and F1",
    ],
    "FlanT5": [
        "Confusion matrix",
        "Class precision, recall, and F1",
    ]
}


# --------------------------------------------------------------------------- #
# UI — all layout in one place, which is the reason to prefer Core here
# --------------------------------------------------------------------------- #
emotion_page = ui.TagList(
ui.layout_columns(
        ui.value_box(
            "Reviews analyzed",
            ui.output_text("selected_review_count"),
        ),
        ui.value_box(
            "Matching business locations",
            ui.output_text("selected_location_count"),
        ),
        ui.value_box(
            "Highest average emotion",
            ui.output_text("highest_emotion_value"),
        ),
        col_widths=[4, 4, 4],
    ),
    ui.card(
        ui.card_header("Average emotion distribution"),
        ui.output_plot(
            "average_emotion_plot",
            height="500px",
        ),
        ui.p(
            ui.output_text("highest_emotion_sentence", inline=True),
            class_="data-note",
        ),
        class_="mb-4",
    ),
    ui.h3("Review-level emotion score distributions", class_="mt-4 mb-3"),
    ui.layout_columns(
        *[
            ui.card(
                ui.card_header(emotion.capitalize()),
                ui.output_plot(
                    f"emotion_hist_{emotion}",
                    height="390px",
                ),
            )
            for emotion in EMOTIONS
        ],
        col_widths=[6, 6, 6, 6, 6, 6, 12],
    ),
)

sentiment_page = ui.TagList(
ui.layout_columns(
        ui.value_box(
            "Reviews analyzed",
            ui.output_text("_flan_selected_review_count"),
        ),
        ui.value_box(
            "Matching business locations",
            ui.output_text("_flan_selected_location_count"),
        ),
        ui.value_box(
            "Overall sentiment score",
            ui.output_text("overall_sentiment"),
        ),
    ui.input_selectize(
        "business", "Search a business:",
        choices=SUMMARY_BUSINESSES, multiple=False,
        options={"placeholder": "type to search..."},
    ),
    ui.input_radio_buttons(
        "mode", "Chart shows:",
        {"count": "Review counts", "pct": "Sentiment %"},
        selected="count",
    ),
    ui.input_select(
        "freq", "Time bucket:",
        {"ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"},
        selected="ME",
    ),
    width=500,
    col_widths=[4, 4, 4],
    ),
    ui.h2("Business Review Explorer"),
    ui.output_ui("overview"),
    ui.output_plot("trend"),
    ui.output_ui("summary_card"),
    ui.output_plot("sentiment_bar_graph"),
)
home_tab = ui.nav_panel(
    "Home",
    ui.tags.style(CSS),
    ui.page_fluid(
        ui.div(
            ui.span("Yelp review dataset analytics", class_="eyebrow"),
            ui.h1("Business dashboard"),
            ui.p(
                "Search for a business name and select the task to view the models analytics' results ",
                class_="lede",
            ),
            class_="mt-3 mb-4",
        ),
        ui.card(
            ui.card_header("Dashboard filters"),
            ui.layout_columns(
                ui.div(
                    ui.input_selectize(
                        "business_name",
                        "Business name",
                        choices=BUSINESS_CHOICES,
                        selected=DEFAULT_BUSINESS,
                        multiple=False,
                        options={
                            "placeholder": "Type a business name...",
                            "maxOptions": 100,
                            "dropdownParent": "body",
                        },
                        width="100%",
                    ),
                    class_="business-filter-column",
                ),
                ui.div(
                    ui.input_select(
                        "task",
                        "Task",
                        choices=["Emotion Classification", "Sentiment Analysis/Summarization"],
                        selected="Emotion Classification",
                        width="100%",
                    ),
                    class_="task-filter-column",
                ),
                ui.navset_hidden(
                    ui.nav_panel("Sentiment Analysis/Summarization", sentiment_page),
                    ui.nav_panel("Emotion Classification",emotion_page),
                    id="page",
                ),
                col_widths=[8, 4],
            ),
            class_="mb-4 dashboard-filters",
        ),

    ),
)


about_tab = ui.nav_panel(
    "About",
    ui.page_fluid(
        ui.div(
            ui.span("Project overview", class_="eyebrow"),
            ui.h1("Yelp Reviews: TLDR"),
            ui.p(
                "An AI-powered Yelp review analytics dashboard that converts large "
                "collections of customer reviews into concise, interactive business insights.",
                class_="lede",
            ),
            class_="mt-3 mb-4",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Project description"),
                ui.p(
                    "Users can search for a business and review an AI-generated summary "
                    "of customer experiences instead of reading hundreds of individual "
                    "reviews. Multiple natural language processing models work together "
                    "to present insights at different levels of detail. What the dashboard provides:"
                ),
                ui.tags.ul(
                    ui.tags.li("Positive, neutral, and negative sentiment analysis"),
                    ui.tags.li("Yearly sentiment trends"),
                    ui.tags.li("Review-level emotion probabilities"),
                     ui.tags.li("Business-level average emotion distribution "),
                    ui.tags.li("Frequently discussed topics and representative keywords"),
                    ui.tags.li("AI-generated review summaries"),
                ),
            ),
            ui.card(
                ui.card_header("Expected users and benefits"),
                ui.h5("Customers"),
                ui.p(
                    "Customers can evaluate businesses more quickly, compare customer "
                    "experiences, and make more informed decisions while spending less "
                    "time manually reviewing feedback."
                ),
                ui.h5("Business owners"),
                ui.p(
                    "Business owners can use the dashboard as an analytics tool to "
                    "understand public perception, identify recurring complaints, and "
                    "recognize areas of strength or potential improvement."
                ),
            ),
            col_widths=[7, 5],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Market research"),
                ui.p(
                    "Related tools include Yelp’s AI-generated review summaries, the "
                    "Yelp Reviews Extractor browser extension, and the open-source Yelp "
                    "Summarization Miner (YUMM). These products generally focus on one "
                    "capability, such as summarization, recommendations, or restaurant-specific "
                    "aspect analysis."
                ),
                ui.p(
                    "Yelp Reviews: TLDR differs by combining sentiment classification, "
                    "emotion analysis, topic discovery, trend visualization, and review "
                    "summarization within one dashboard for multiple types of businesses."
                ),
            ),
            ui.card(
                ui.card_header("Key implementation challenges"),
                ui.tags.ul(
                    ui.tags.li("Interpreting mixed opinions, context, and sarcasm"),
                    ui.tags.li("Distinguishing between closely related emotions"),
                    ui.tags.li("Generating summaries that remain faithful to the reviews"),
                    ui.tags.li("Identifying coherent and non-repetitive topics"),
                    ui.tags.li("Processing a large review dataset efficiently"),
                    ui.tags.li("Evaluating each NLP component with appropriate metrics"),
                ),
            ),
            col_widths=[7, 5],
        ),
    ),
)

data_tab = ui.nav_panel(
    "Data",
    ui.page_fluid(
        ui.div(
            ui.span("Dataset and preparation", class_="eyebrow"),
            ui.h1("Yelp Open Dataset"),
            ui.p(
                "Source data, preprocessing decisions, and the cleaned California "
                "review sample used by the dashboard.",
                class_="lede",
            ),
            class_="mt-3 mb-4",
        ),
        ui.card(
            ui.card_header("Yelp Open Dataset"),
            ui.div(
                ui.p(
                    "We built our tool using the Yelp Open Dataset "
                    "available through Kaggle (",
                    ui.a(
                        "https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/",
                        href="https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/",
                        target="_blank",
                        rel="noopener",
                    ),
                    "). The dataset is a subset of Yelp’s business, review, and "
                    "user data and contains information about businesses across "
                    "major metropolitan areas in the United States and Canada. "
                    "For this project, we will only use the two JSON files "
                    "“yelp_academic_dataset_review.json”, which contains data on "
                    "6,990,280 reviews, and “yelp_academic_dataset_business.json”, "
                    "which contains data on 150,346 businesses. The data cover "
                    "approximately 2005 through the beginning of 2022 and include "
                    "businesses from 13 U.S. states and two Canadian provinces, "
                    "such as Pennsylvania, Florida, Tennessee, Louisiana, Indiana, "
                    "Nevada, Arizona, Alberta, and British Columbia."
                ),
                class_="content-copy",
            ),
            class_="section-card",
        ),
        ui.card(
            ui.card_header("Data preprocessing"),
            ui.div(
                ui.p(
                    "First of all, the review and business datasets were merged "
                    "using their shared “business_id” column in order to connect "
                    "each review to information such as the business’s name. The "
                    "primary input for the NLP models is the ‘text’ column "
                    "from the review dataset, which is the customer’s actual "
                    "comment. The dataset also includes a “stars” column containing "
                    "each review’s numerical rating on a scale from 1 to 5 stars. "
                    "Data preprocessing included data cleaning techniques "
                    "such as removing duplicate and missing reviews’ text, "
                    "standardizing text formatting, handling unusual characters, "
                    "as well as removing irrelevant columns, and processing the "
                    "date column."
                ),
                ui.p(
                    "For sentiment classification, the review’s star rating was "
                    "transformed into a reference sentiment label. Reviews with "
                    "ratings > 3 stars were labeled ‘positive’, 3-star reviews "
                    "were labeled ‘neutral’, and reviews with ratings < 3 stars "
                    "were labeled ‘negative’. The labeled reviews were "
                    "divided into training, validation, and test sets using "
                    "stratified sampling to preserve the relative frequency of each "
                    "sentiment class. The training set was used to fine-tune the "
                    "models, the validation set supported model selection and "
                    "hyperparameter tuning, and the test set was reserved for "
                    "final performance evaluation."
                ),
                ui.h5("Additional filtering applied"),
                ui.tags.ul(
                    ui.tags.li(
                        "Only English-language reviews were retained using the "
                        "fastText language-identification model."
                    ),
                    ui.tags.li(
                        "Only businesses located in California (CA) were retained."
                    ,
                    ui.tags.ul(
                    ui.tags.li("Keeps the scope computationally manageable."),
                    ui.tags.li("Still provides enough reviews and businesses."),
                    ui.tags.li("Supports focused dashboard development."),
                    ),
                    ),
                ),
                class_="content-copy",
            ),
            class_="section-card",
        ),
        ui.p("The data is shown in the table below:", class_="data-note"),
        ui.card(
            ui.output_data_frame("data_table"),
            class_="mb-4",
        ),
    ),
)

models_tab = ui.nav_panel(
    "Models",
    ui.page_fluid(
        ui.div(
            ui.span("Model evaluation", class_="eyebrow"),
            ui.h1("Model performance"),
            ui.p(
                "Explore evaluation metrics, diagnostic charts, and "
                "class-specific results for the trained NLP models.",
                class_="lede",
            ),
            class_="mt-3 mb-4",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "model",
                    "Model",
                    choices=list(RESULTS),
                    selected="DistilBERT",
                ),
                ui.input_select(
                    "chart",
                    "Chart",
                    choices=MODEL_CHARTS["DistilBERT"],
                    selected=MODEL_CHARTS["DistilBERT"][0],
                ),
                ui.input_switch("annotate", "Annotate values", True),
                ui.hr(),
                width=280,
            ),
            ui.output_ui("metric_boxes"),
            ui.card(
                ui.card_header("Selected chart"),
                ui.output_plot("model_plot", height="520px"),
            ),
            ui.card(
                ui.card_header("Class-specific metrics"),
                ui.output_data_frame("class_metrics"),
            ),
            ui.card(
                ui.card_header("Model comparison summary"),
                ui.output_data_frame("summary"),
            ),
        ),
    ),
)

github_tab = ui.nav_panel(
    "GitHub",
    ui.div(
        ui.span("Source", class_="eyebrow"),
        ui.h1("Where the code lives"),
        ui.p(
            "Data preparation, model training, and this application are maintained "
            "in one repository.",
            class_="lede",
        ),
        class_="mt-3 mb-4",
    ),
    ui.card(
        ui.card_header("Repository"),
        ui.p(ui.a(REPO_URL, href=REPO_URL, target="_blank", rel="noopener")),
        ui.markdown(f"```\ngit clone {REPO_URL}.git\n```"),
        ui.markdown(
            "- `src/datafiles/` — data storage\n"
            "- `src/models/` — training scripts and saved artifacts\n"
            "- `src/preprocessing/` — data preprocessing script\n"
            "- `shiny_project/test.py` — this dashboard\n"
        ),
    ),
)

app_ui = ui.page_navbar(
    home_tab,
    about_tab,
    data_tab,
    models_tab,
    github_tab,
    title="Yelp:TLDR",
    id="tabs",
    fillable=False,
)

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
def server(input, output, session):
    # ---- Home tab: selected-business emotion analytics ---------------------
    @reactive.calc
    def selected_business_name() -> str:
        selected = input.business_name()
        return selected.strip() if selected else DEFAULT_BUSINESS

    @reactive.calc
    def selected_business_summary() -> dict:
        business_name = selected_business_name()

        selected = EMOTION_SUMMARY.filter(
            pl.col("business_name") == business_name
        )

        if selected.is_empty():
            raise ValueError(
                f"No precomputed business-level emotion summary was found "
                f"for '{business_name}'."
            )

        return selected.row(0, named=True)

    @reactive.calc
    def reactive_selected_business_summary() -> dict:
        business_name = input.business()

        selected = EMOTION_SUMMARY.filter(
            pl.col("business_name") == business_name
        )

        if selected.is_empty():
            raise ValueError(
                f"No precomputed business-level emotion summary was found "
                f"for '{business_name}'."
            )

        return selected.row(0, named=True)
    @reactive.calc
    def selected_review_overall_sentiment():
        business_name = input.business()
        business_sentiments = FLAN_SENTIMENT[FLAN_SENTIMENT["business_name"] == business_name]
        label_counts = business_sentiments["prediction_sentiment"].value_counts()
        return label_counts.idxmax()

    @reactive.calc
    def selected_review_emotions() -> pl.DataFrame:
        business_name = selected_business_name()

        selected = (
            EMOTION_REVIEWS_LAZY
            .filter(pl.col("business_name") == business_name)
            .select(EMOTIONS)
            .collect()
        )

        if selected.is_empty():
            raise ValueError(
                f"No precomputed review-level emotion scores were found "
                f"for '{business_name}'."
            )

        return selected
    @reactive.calc
    def all_business_reviews():
        """
        This returns all reviews not just the overall sentiment.
        """
        business_name = input.business()
        if not business_name:
            return None
        return FLAN_SENTIMENT[FLAN_SENTIMENT["business_name"] == business_name]

    @render.text
    def selected_review_count():
        return f"{int(selected_business_summary()['review_count']):,}"
    @render.text
    def _flan_selected_review_count():
        return f"{int(reactive_selected_business_summary()['review_count']):,}"

    @render.text
    def selected_location_count():
        return (
            f"{int(selected_business_summary()['matching_business_id_count']):,}"
        )
    @render.text
    def _flan_selected_location_count():
        return f"{int(reactive_selected_business_summary()['matching_business_id_count']):,}"

    @render.text
    def overall_sentiment():
        return selected_review_overall_sentiment()

    @render.ui
    def overview():
        business_name = input.business()
        if not business_name:
            return ui.p("Select a business from the sidebar to see its sentiment "
                        "trend and summary.")
        sub = all_business_reviews()
        return ui.tags.div(
            ui.h3(business_name),
            ui.p(f"{len(sub)} reviews on record "
                 f"({sub['date'].min().date()} to {sub['date'].max().date()})"),
        )
    @render.plot
    def sentiment_bar_graph():
        all_reviews = all_business_reviews()
        counts = (all_reviews["prediction_sentiment"].value_counts().reindex(SENTIMENT_LABELS, fill_value=0))
        fig,ax = plt.subplots(figsize=(5,4))
        ax.bar(counts.index,counts.values,color=[SENTIMENT_COLORS[i] for i in counts.index])
        ax.set_ylabel("number of reviews")
        ax.set_title(f"Sentiment distribution — {input.business()}")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom")
        fig.tight_layout()
        return fig

    @render.plot
    def trend():
        sub =  all_business_reviews()
        if sub is None or sub.empty:
            return None
        freq = _safe_freq(input.freq())
        monthly = (sub.set_index("date")
                   .groupby([pd.Grouper(freq=freq), "prediction_sentiment"])
                   .size().unstack(fill_value=0))
        # keep a stable label order + colors
        for lab in SENTIMENT_LABELS:
            if lab not in monthly.columns:
                monthly[lab] = 0
        monthly = monthly[SENTIMENT_LABELS]

        if input.mode() == "pct":
            totals = monthly.sum(axis=1).replace(0, 1)
            monthly = monthly.div(totals, axis=0) * 100

        fig, ax = plt.subplots(figsize=(9, 4.5))
        monthly.plot(kind="area", stacked=True, ax=ax,
                     color=[SENTIMENT_COLORS[l] for l in SENTIMENT_LABELS])
        ax.set_xlabel("date")
        ax.set_ylabel("% of reviews" if input.mode() == "pct" else "reviews")
        ax.set_title(f"Predicted sentiment over time — {input.business()}")
        ax.legend(title="sentiment", loc="upper left")
        fig.tight_layout()
        return fig

    @render.ui
    def summary_card():
        business_name = input.business()
        if not business_name:
            return None
        row = FLAN_SUMMARY[FLAN_SUMMARY["business_name"] == business_name]
        if row.empty:
            return ui.tags.div(
                ui.h4("Summary"),
                ui.p(ui.em("No summary available for this business "
                           "(too few reviews to summarize).")),
            )
        r = row.iloc[0]
        return ui.tags.div(
            ui.h4("Overall summary"),
            ui.p(r["summary"]),
            ui.p(ui.tags.small(
                f"Based on {int(r['n_reviews'])} reviews — "
                f"{r["sentiment_distro"]}%, "
            )),
            style="background:#f6f6f6; padding:12px; border-radius:8px; margin-top:12px;",
        )
    @render.text
    def highest_emotion_value():
        row = selected_business_summary()
        emotion = max(EMOTIONS, key=lambda name: float(row[name]))
        return emotion.capitalize()

    @render.text
    def highest_emotion_sentence():
        row = selected_business_summary()
        emotion = max(EMOTIONS, key=lambda name: float(row[name]))
        return (
            f"Emotion with highest average score: {emotion} "
            f"({float(row[emotion]):.4f})"
        )

    @render.plot
    def average_emotion_plot():
        return plot_average_emotion_distribution(
            selected_business_summary(),
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_anger():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "anger",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_disgust():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "disgust",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_fear():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "fear",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_joy():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "joy",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_neutral():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "neutral",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_sadness():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "sadness",
            selected_business_name(),
        )

    @render.plot
    def emotion_hist_surprise():
        return plot_emotion_histogram(
            selected_review_emotions(),
            "surprise",
            selected_business_name(),
        )

    @reactive.effect
    def _sync_page():
        ui.update_navs("page", selected=input.task())
    # ---- Models tab --------------------------------------------------------
    @reactive.calc
    def current():
        return RESULTS[input.model()]

    @reactive.effect
    def update_chart_choices():
        model_name = input.model()
        choices = MODEL_CHARTS[model_name]
        ui.update_select(
            "chart",
            choices=choices,
            selected=choices[0],
        )

    @render.plot
    def model_plot():
        return PLOTTERS[input.chart()](
            current(),
            annotate=input.annotate(),
        )

    @render.ui
    def metric_boxes():
        res = current()

        if res["kind"] == "distilbert" or res["kind"] == "flan-t5":
            return ui.layout_columns(
                ui.value_box("Accuracy", f"{res['accuracy']:.1%}"),
                ui.value_box("Macro F1", f"{res['f1_macro']:.3f}"),
                ui.value_box("Weighted F1", f"{res['f1_weighted']:.3f}"),
                ui.value_box("Test loss", f"{res['test_loss']:.3f}"),
                col_widths=[3, 3, 3, 3],
            )

        fpr, tpr = roc_points(res["y_true"], res["y_score"])
        return ui.layout_columns(
            ui.value_box("AUC", f"{auc(fpr, tpr):.3f}"),
            ui.value_box("Accuracy", f"{binary_accuracy(res):.1%}"),
            ui.value_box(
                "Positive rate",
                f"{res['y_true'].mean():.1%}",
            ),
            col_widths=[4, 4, 4],
        )

    @render.data_frame
    def class_metrics():
        res = current()

        if res["kind"] == "distilbert" or res["kind"] == "flan-t5":
            table = res["class_metrics"].copy()
            for column in ["Precision", "Recall", "F1"]:
                table[column] = table[column].map(lambda value: f"{value:.3f}")
            table["Support"] = table["Support"].map(lambda value: f"{int(value):,}")
        else:
            table = pd.DataFrame(
                {
                    "Information": [
                        "Class-specific metrics are currently available "
                        "for DistilBERT, and Flan-T5.",
                    ]
                }
            )

        return render.DataGrid(table, width="100%")

    @render.data_frame
    def summary():
        rows = []

        for name, res in RESULTS.items():
            if res["kind"] == "distilbert" or res["kind"] == "flan-t5":
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": round(res["accuracy"], 3),
                        "Macro precision": round(res["precision_macro"], 3),
                        "Macro recall": round(res["recall_macro"], 3),
                        "Macro F1": round(res["f1_macro"], 3),
                        "Weighted F1": round(res["f1_weighted"], 3),
                        "Test loss": round(res["test_loss"], 3),
                    }
                )
            else:
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": round(binary_accuracy(res), 3),
                        "Macro precision": None,
                        "Macro recall": None,
                        "Macro F1": None,
                        "Weighted F1": None,
                        "Test loss": round(float(res["val_loss"].min()), 3),
                    }
                )

        return render.DataGrid(pd.DataFrame(rows), width="100%")

    @render.data_frame
    def data_table():
        '''
        Currently only loading in the first 1000 entires, we can work out a pagination schema leveraging
        polaris, and offsets if we want
        :return:
        '''
        return render.DataGrid(LOCAL_PATH_DATA.head(10000), width="100%", height="1000px", filters=True)
app = App(app_ui, server, static_assets=WWW)