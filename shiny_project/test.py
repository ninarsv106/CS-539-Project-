
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

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
"""

# --------------------------------------------------------------------------- #
# Tab 1 data — STUB. Replace the body of load_businesses() with your loader.
#
# Expected columns (mirrors the Yelp Open Dataset business schema, plus two
# fields you derive yourself: `photo` and `blurb`):
#
#   business_id  name  address  city  state  stars  review_count
#   categories   comma-separated string, e.g. "Coffee & Tea, Breakfast"
#   photo        filename that exists in ./www/  (or "" for no image)
#   blurb        your summary text for the card
#
# Real version will look roughly like:
#
#   biz = pd.read_json("data/yelp_academic_dataset_business.json", lines=True)
#   pho = (pd.read_json("data/photos.json", lines=True)
#            .drop_duplicates("business_id")
#            .set_index("business_id")["photo_id"])
#   biz["photo"] = biz["business_id"].map(pho).fillna("") + ".jpg"
#   biz["blurb"] = your_review_summaries(biz["business_id"])
#   return biz.loc[biz["is_open"] == 1, COLUMNS].reset_index(drop=True)
#
# This runs once at process startup, not per session — which is exactly where
# you want a multi-hundred-MB JSON read to happen.
# --------------------------------------------------------------------------- #
COLUMNS = ["business_id", "name", "address", "city", "state",
           "stars", "review_count", "categories", "photo", "blurb"]

_STUB_ROWS = [
    ("stub-0001", "Sunrise Diner", "1420 S 9th St", "Philadelphia", "PA",
     4.5, 812, "Diners, Breakfast & Brunch, American (Traditional)",
     "sunrise_diner.jpg",
     "Reviewers keep coming back for the scrapple and the counter seating. "
     "Weekend waits run long; service is the most praised attribute."),
    ("stub-0002", "Peak Cycle Works", "3311 E Speedway Blvd", "Tucson", "AZ",
     4.0, 137, "Bike Repair/Maintenance, Sporting Goods, Bikes",
     "peak_cycle.jpg",
     "Small shop with a strong repair reputation. Sentiment dips around "
     "pricing on new builds but stays high on tune-ups."),
    ("stub-0003", "Bayou & Vine", "728 Magazine St", "New Orleans", "LA",
     3.5, 2043, "Cajun/Creole, Cocktail Bars, Seafood",
     "bayou_vine.jpg",
     "High volume, polarised reviews. Food scores well, wait times and noise "
     "drive most of the one-star text."),
    ("stub-0004", "Nine Bells Tea House", "1900 8th Ave S", "Nashville", "TN",
     4.5, 61, "Coffee & Tea, Bakeries, Cafes",
     "nine_bells.jpg",
     "Newer listing with few but consistent reviews. Quiet-workspace "
     "mentions dominate the text."),
]

REPO_URL = "https://github.com/ninarsv106/CS-539-Project-"
def load_businesses() -> pd.DataFrame:
    """STUB — swap this body for the real Kaggle Yelp load."""
    return pd.DataFrame(_STUB_ROWS, columns=COLUMNS)


BUSINESSES = load_businesses()

# Fields the search bar looks at, and a lowercased haystack built once at
# startup so filtering stays cheap when this is 150k rows instead of 4.
SEARCH_FIELDS = ["name", "city", "state", "categories", "blurb"]
HAYSTACK = (BUSINESSES[SEARCH_FIELDS].astype(str)
            .agg(" ".join, axis=1).str.lower())

MAX_CARDS = 24  # cap on cards rendered at once
LOCAL_PATH_DATA = pl.read_csv(Path(__file__).parent / ".." / "datafiles" / "yelp_reviews_clean_CA.csv")
WWW = Path(__file__).parent / "www"
WWW.mkdir(exist_ok=True)  # static_assets needs the directory to exist

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


RESULTS = {
    "DistilBERT": DISTILBERT_RESULTS,
    "Logistic regression": _fake_results(1, 0.55),
    "Random forest": _fake_results(2, 0.75),
    "Gradient boosting": _fake_results(3, 0.88),
    "Neural net (MLP)": _fake_results(4, 0.80),
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
    if res["kind"] == "distilbert":
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
    if res["kind"] != "distilbert":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(
            0.5,
            0.5,
            "Class-specific precision, recall, and F1\n"
            "are available for the DistilBERT model.",
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
    "Logistic regression": [
        "ROC curve",
        "Confusion matrix",
        "Training and validation loss",
    ],
    "Random forest": [
        "ROC curve",
        "Confusion matrix",
        "Training and validation loss",
    ],
    "Gradient boosting": [
        "ROC curve",
        "Confusion matrix",
        "Training and validation loss",
    ],
    "Neural net (MLP)": [
        "ROC curve",
        "Confusion matrix",
        "Training and validation loss",
    ],
}


# --------------------------------------------------------------------------- #
# UI — all layout in one place, which is the reason to prefer Core here
# --------------------------------------------------------------------------- #
def image_or_placeholder(filename: str, alt: str):
    if filename and (WWW / filename).exists():
        return ui.img(src=filename, alt=alt, class_="block-img")
    return ui.div(filename or "no photo", class_="block-img placeholder")


def stars_glyphs(stars: float) -> str:
    full = int(stars)
    return "\u2605" * full + ("\u00bd" if stars - full >= 0.5 else "")


def business_card(row) -> ui.Tag:
    """One summary block. `row` is a namedtuple from DataFrame.itertuples()."""
    cats = [c.strip() for c in str(row.categories).split(",") if c.strip()][:4]
    return ui.card(
        ui.layout_columns(
            ui.div(
                ui.span(f"{row.city}, {row.state}", class_="eyebrow"),
                ui.h4(row.name),
                ui.p(
                    ui.span(stars_glyphs(row.stars), class_="stars"),
                    ui.span(f"{row.stars:.1f} \u00b7 {row.review_count:,} reviews",
                            class_="hitcount"),
                ),
                ui.p(row.blurb),
                ui.div(*[ui.span(c, class_="chip") for c in cats]),
            ),
            image_or_placeholder(row.photo, row.name),
            col_widths=[8, 4],
        ),
        class_="mb-3",
    )

home_tab = ui.nav_panel(
    "Home",
    ui.tags.style(CSS),
    ui.div(
        ui.span("Yelp review dataset", class_="eyebrow"),
        ui.h1("Businesses in the sample"),
        ui.p(
            "Search by name, city or category. Model results for these "
            "businesses are available on the Models tab.",
            class_="lede",
        ),
        class_="mt-3 mb-4",
    ),
    ui.layout_columns(
        ui.input_text("search", None, width="100%",
                      placeholder="Search businesses, cities or categories…"),
        ui.output_text("hits", inline=True),
        col_widths=[6, 6],
    ),
    ui.output_ui("blocks"),
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
    # ---- Tab 1: search over BUSINESSES ------------------------------------
    @reactive.calc
    def query():
        return (input.search() or "").strip().lower()

    @reactive.calc
    def matches() -> pd.DataFrame:
        q = query()
        if not q:
            return BUSINESSES
        return BUSINESSES[HAYSTACK.str.contains(q, regex=False)]

    @render.text
    def hits():
        n = len(matches())
        shown = min(n, MAX_CARDS)
        if not query():
            return f"{len(BUSINESSES):,} businesses \u00b7 showing {shown}"
        return f"{n:,} match \u00b7 showing {shown}"

    @render.ui
    def blocks():
        found = matches()
        if found.empty:
            return ui.card(ui.p(f"No business matches \u201c{input.search()}\u201d. "
                                "Try a city or a category instead."))
        return [business_card(r) for r in found.head(MAX_CARDS).itertuples()]

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

        if res["kind"] == "distilbert":
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

        if res["kind"] == "distilbert":
            table = res["class_metrics"].copy()
            for column in ["Precision", "Recall", "F1"]:
                table[column] = table[column].map(lambda value: f"{value:.3f}")
            table["Support"] = table["Support"].map(lambda value: f"{int(value):,}")
        else:
            table = pd.DataFrame(
                {
                    "Information": [
                        "Class-specific metrics are currently available "
                        "for DistilBERT."
                    ]
                }
            )

        return render.DataGrid(table, width="100%")

    @render.data_frame
    def summary():
        rows = []

        for name, res in RESULTS.items():
            if res["kind"] == "distilbert":
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