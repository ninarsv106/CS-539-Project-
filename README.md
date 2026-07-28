# CS-539-Project-
Website link: https://konstantina-rasvani.shinyapps.io/yelp-tldr1/
### Installing the Environment
1. `python -m venv .venv`
2.  For Linux `source .venv/bin/activate`
3.  For Windows `.venv\Scripts\Activate.ps1`
4.  `pip install -r requirements.txt`

### Getting the Data
1. Download the data from the dataset link in `datafiles/Dataset_download_link.txt`
2. Place `yelp_reviews_clean_CA.csv` inside the folder `datafiles`
3. Download the files from the links in `datafiles/yelp_reviews_with_hf_emotions_download_link.txt` and `datafiles/yelp_all_business_emotion_summary_download_link.txt`
4. Place `yelp_reviews_with_hf_emotions.csv` and `yelp_all_business_emotion_summary.csv` inside the folder `datafiles`

### To run the Shiny app locally for testing, follow these steps
1. `pip install shiny pandas numpy matplotlib polars`
1. Navigate to  `shiny_project/`
2.  Then run on the command line `shiny run --reload --launch-browser test.py`

### Model Location:
1. The trained `flan-t5-base` model has been uploaded to `https://huggingface.co/SSDevForge/flan-t5-base`
