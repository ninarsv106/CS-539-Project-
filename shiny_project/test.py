
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
# Model artifacts. Top-level code in Core runs ONCE per process, not per
# session — this is where real loading belongs, e.g.
#     RESULTS = {name: joblib.load(f"artifacts/{name}.pkl") for name in ...}
# --------------------------------------------------------------------------- #
def _fake_results(seed: int, skill: float) -> dict:
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.binomial(1, 0.3, n)
    score = np.clip(rng.normal(0.5 + skill * (y - 0.5), 0.18), 0, 1)
    epochs = np.arange(1, 26)
    return {
        "y_true": y,
        "y_score": score,
        "features": [f"feature_{i:02d}" for i in range(10)],
        "importance": np.sort(rng.dirichlet(np.ones(10) * 0.7))[::-1],
        "train_loss": 0.7 * np.exp(-epochs / 7) + 0.08,
        "val_loss": 0.7 * np.exp(-epochs / 9) + 0.12 + rng.normal(0, 0.008, 25),
        "epochs": epochs,
    }


RESULTS = {
    "Logistic regression": _fake_results(1, 0.55),
    "Random forest": _fake_results(2, 0.75),
    "Gradient boosting": _fake_results(3, 0.88),
    "Neural net (MLP)": _fake_results(4, 0.80),
}


# --------------------------------------------------------------------------- #
# Metrics helpers (no sklearn needed for the demo)
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
    return np.array([[np.sum((y_true == t) & (pred == p)) for p in (0, 1)] for t in (0, 1)])


def _frame(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dfe4e7")


def plot_roc(res, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fpr, tpr = roc_points(res["y_true"], res["y_score"])
    ax.plot(fpr, tpr, color=ACCENT, lw=2.2)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#c3ccd1")
    if annotate:
        ax.annotate(f"AUC {auc(fpr, tpr):.3f}", (0.62, 0.22),
                    fontsize=13, fontweight="bold", color=INK)
    _frame(ax, "ROC curve", "False positive rate", "True positive rate")
    return fig


def plot_confusion(res, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cm = confusion(res["y_true"], res["y_score"])
    ax.imshow(cm / cm.sum(), cmap="BuGn", vmin=0)
    ax.set_xticks([0, 1], ["pred: stay", "pred: churn"])
    ax.set_yticks([0, 1], ["true: stay", "true: churn"])
    if annotate:
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=SURFACE if v > cm.max() * 0.6 else INK)
    _frame(ax, "Confusion matrix at threshold 0.50", "", "")
    ax.tick_params(labelsize=9)
    return fig


def plot_importance(res, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(res["features"]))
    ax.barh(y, res["importance"], color=ACCENT, height=0.62)
    ax.set_yticks(y, res["features"], fontsize=9)
    ax.invert_yaxis()
    if annotate:
        for yi, v in zip(y, res["importance"]):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    _frame(ax, "Permutation importance", "Mean decrease in AUC", "")
    return fig


def plot_learning_curve(res, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["epochs"], res["train_loss"], color=ACCENT, lw=2, label="train")
    ax.plot(res["epochs"], res["val_loss"], color=WARM, lw=2, label="validation")
    if annotate:
        best = int(np.argmin(res["val_loss"]))
        ax.axvline(res["epochs"][best], color="#c3ccd1", ls="--", lw=1)
        ax.text(res["epochs"][best] + 0.4, res["val_loss"].max(),
                f"best epoch {res['epochs'][best]}", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    _frame(ax, "Learning curve", "Epoch", "Log loss")
    return fig


PLOTTERS = {
    "ROC curve": plot_roc,
    "Confusion matrix": plot_confusion,
    "Feature importance": plot_importance,
    "Learning curve": plot_learning_curve,
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
            ui.input_select("model", "Model", choices=list(RESULTS)),
            ui.input_select("chart", "Chart", choices=list(PLOTTERS)),
            ui.input_switch("annotate", "Annotate values", True),
            ui.hr(),
            ui.markdown("Charts are produced by functions in `PLOTTERS`."),
            width=280,
        ),
        ui.layout_columns(
            ui.value_box("AUC", ui.output_text("vb_auc")),
            ui.value_box("Accuracy", ui.output_text("vb_acc")),
            ui.value_box("Positive rate", ui.output_text("vb_rate")),
            col_widths=[4, 4, 4],
        ),
        ui.card(ui.output_plot("model_plot", height="460px")),
        ui.card(ui.card_header("All models"), ui.output_data_frame("summary")),
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

    # ---- Tab 2: dropdown-driven plots ------------------------------------
    @reactive.calc
    def current():
        return RESULTS[input.model()]

    @render.plot
    def model_plot():
        # look the function up, call it — nothing Shiny-specific in the plotters
        return PLOTTERS[input.chart()](current(), annotate=input.annotate())

    @render.text
    def vb_auc():
        return f"{auc(*roc_points(current()['y_true'], current()['y_score'])):.3f}"

    @render.text
    def vb_acc():
        cm = confusion(current()["y_true"], current()["y_score"])
        return f"{np.trace(cm) / cm.sum():.1%}"

    @render.text
    def vb_rate():
        return f"{current()['y_true'].mean():.1%}"

    @render.data_frame
    def summary():
        rows = []
        for name, res in RESULTS.items():
            cm = confusion(res["y_true"], res["y_score"])
            rows.append({
                "Model": name,
                "AUC": round(auc(*roc_points(res["y_true"], res["y_score"])), 3),
                "Accuracy": round(float(np.trace(cm) / cm.sum()), 3),
                "Best val loss": round(float(res["val_loss"].min()), 3),
            })
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