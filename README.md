# CS-539-Project-
### Installing the Environment
1. `python -m venv .venv`
2.  For Linux `source .venv/bin/activate`
3.  For Windows `.venv\Scripts\Activate.ps1`
4.  `pip install -r requirements.txt`

### Getting the Data
1. Download the data from the dataset link in `datafiles/Dataset_download_link.txt`
2. Place the `yelp_reviews_clean_CA.csv` in the folder `datafiles/yelp_reviews_clean_CA.csv`

### To run the Shiny app locally for testing, follow these steps
1. `pip install shiny pandas numpy matplotlib polars`
1. Navigate to  `shiny_project/`
2.  Then run on the command line `shiny run --reload --launch-browser test.py`
