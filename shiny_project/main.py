
from shiny import App, reactive, render, ui
from formatted_txt import overview, footer
'''
Defining UI Segment Blocks here, leveraging the "Core" Syntax
'''
home_ui =  ui.input_submit_textarea(
        "search",
        label="Search Local Businesses",
        placeholder="Enter Business Name Here ...",
        rows=2,
        toolbar=[
            ui.input_action_button("clear", "Clear", class_="btn-sm btn-danger"),
        ],
    )
about_ui =ui.card(
        ui.card_header("Welcome to Yelp:TLDR!",
                       class_="text-center fs-5 fw-semibold text-uppercase bg-light"),
        ui.p(overview),
        ui.card_footer(footer),
        full_screen=True,
    )

repo_ui = ui.card(
  ui.card_header("A header",
  ),
 ui. card_body(
    ui.markdown("Checkout our codebase! [Repo](https://github.com/ninarsv106/CS-539-Project-)")
  )
)

model_ui = ui.card(
    ui.card_header("Model Overview", class_="text-center fw-semibold"),
    ui.markdown(
        """
The models that were used for this project are the following:
1. DistilBERT
1. Flan-T5 (Base)
1. DistilRoBERTa
1. BertTopic
1. FastTopic

### Model Grading Criteria:
- `accuracy` — overall hit rate
- `ROC AUC` — ranking quality, insensitive to threshold
- `Placeholder for more metrics`

        """
    ),
    ui.card_footer("Metrics require a label column.", class_="small text-muted"),
)

page_ui = ui.page_navbar(ui.nav_panel("Home", home_ui),
    ui.nav_panel("About", about_ui),
    ui.nav_panel("Model Overview", model_ui),
    ui.nav_panel("Repo", repo_ui),
    title = "Yelp:TLDR",
    id="page_ui",
)

'''
Below here is where all of the interactive logic will go to implement features behind the buttons
'''
def server(input):
    pass

'''
Calling Object below
'''
app = App(page_ui,server)