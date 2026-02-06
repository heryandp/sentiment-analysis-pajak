# Sentiment Analysis Pajak

End-to-end experiments for Indonesian tax sentiment analysis using Google News data, YouTube comments, classic ML models, and an LLM-assisted labeling pipeline. The project is organized around Jupyter notebooks that implement each stage of the workflow.

## Project Flow (High Level)
1. **Collect news** from Google News via Serper API, clean dates, deduplicate, and extract full content.
2. **Convert content** JSON into a single CSV for modeling.
3. **Preprocess text** (cleaning, stopword removal, stemming) and train NB/SVM baselines.
4. **LLM labeling** (Gemma 3 via API) to generate sentiment labels, then train NB/SVM and compare.
5. **Optional deep learning** with Indo-RoBERTa fine-tuning and evaluation.
6. **YouTube pipeline** for comment scraping and sentiment modeling on video comments.

## Repository Structure
- `collect.ipynb`: Google News scraping + cleaning + aggregation + content extraction
- `convert_json_to_csv.py`: Flatten `raw_content/*.json` into `combined_data.csv`
- `sentimen.ipynb`: Preprocessing + classical ML (NB/SVM) on news dataset
- `llm.ipynb`: LLM labeling + model training + evaluation + model export
- `sentimen_yt.ipynb`: YouTube comments scraping + preprocessing + ML baseline
- `example/`: Example outputs
- `*.png`: Evaluation visuals

## Data & Artifacts
Large datasets and model artifacts are excluded from git. See `.gitignore` for excluded paths. You should store large files locally or in separate storage.

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Notebook Details

### 1) `collect.ipynb` (News Collection)
**Purpose**: Collect Google News articles about Indonesian tax topics and extract full content.

**Flow**:
- Set `API_KEY` for Serper.
- Query Serper News endpoint and store raw responses in `raw/`.
- Normalize dates and write cleaned JSON into `raw_clean_date/`.
- Filter out `pajak.go.id` links.
- Aggregate and deduplicate by `link` into `aggregated_news.json`.
- Fetch full article content via Serper scrape endpoint into `raw_content/`.

**Key outputs**:
- `raw/*.json`
- `raw_clean_date/*.json`
- `aggregated_news.json`
- `raw_content/*.json`

### 2) `convert_json_to_csv.py` (Flatten Content)
**Purpose**: Merge `raw_content/*.json` into a single dataset.

**Output**:
- `combined_data.csv` with selected fields: `title`, `link`, `snippet`, `date_clean`, `full_content`

### 3) `sentimen.ipynb` (Baseline ML on News)
**Purpose**: Clean and normalize text, then train Naive Bayes and SVM classifiers.

**Key steps**:
- Text cleaning and normalization
- Stopword removal and stemming (Sastrawi)
- TF-IDF vectorization
- Train and evaluate Naive Bayes and SVM
- Confusion matrices and classification reports

**Key outputs**:
- `combined_data_cleaned.csv`
- `combined_data_stemmed.csv`
- `combined_data_sentiment.csv`
- `testing_manual_results.csv`

### 4) `llm.ipynb` (LLM Labeling + Model Training)
**Purpose**: Use LLM to label sentiment and train/compare models.

**Key steps**:
- Load `combined_data.csv` and create labels via LLM API (Gemma 3)
- Train NB and SVM baselines on labeled data
- Evaluate and compare models (accuracy, confusion matrix)
- Export vectorizer and model
- Optional fine-tuning/evaluation with Indo-RoBERTa
- Additional experimentation with data injection for performance improvement

**Key outputs**:
- `combined_data_labeled.csv`
- `sample_labeled_result.csv`
- `model_comparison.png`
- `tfidf_vectorizer.pkl`
- `sentiment_model_final.pkl`
- Optional deep learning checkpoints in local folders

### 5) `sentimen_yt.ipynb` (YouTube Comments)
**Purpose**: Scrape comments from a YouTube video and build a sentiment model.

**Key steps**:
- Download YouTube comments via `youtube-comment-downloader`
- Normalize slang using the colloquial lexicon
- Clean, stem, and vectorize text
- Train NB and SVM baselines

**Key outputs**:
- `youtube_data.csv` / `youtube_data.json` / `youtube_data.html`
- `yt_cleaned_full_normalized.csv`
- `yt_cleaned_sentimen.csv`

## Running Order (Recommended)
1. `collect.ipynb`
2. `convert_json_to_csv.py`
3. `sentimen.ipynb`
4. `llm.ipynb`
5. `sentimen_yt.ipynb`

## Environment Variables
- `API_KEY`: Serper API key used in `collect.ipynb` for News + Scrape endpoints (set in `.env` or environment).


## Sample App (API + UI)
A simple Flask demo lives in `sample_app/`.

Run it:
```bash
cd sample_app
python app.py
```
Then open `http://localhost:8000`.


Environment (optional):
- `MODEL_PATH`: path to `sentiment_model_final.pkl`
- `VECTORIZER_PATH`: path to `tfidf_vectorizer.pkl`
- `EVAL_DATA_PATH`: CSV for evaluation report (default `testing_manual_results.csv`)
- `EVAL_TEXT_COL`: text column in eval CSV (default `processed_full_content`)
- `EVAL_LABEL_COL`: label column in eval CSV (default `label_manual`)
- `PORT`: server port (default 8000)

## Notes & Warnings
- `collect.ipynb` reads the API key from environment variable `API_KEY`.
- Large datasets/models are excluded from git; recreate them by running the notebooks.
- Notebooks are the main source of truth for the workflow.

## Citation / Data Sources
- Google News via Serper API
- Serper Scrape API for full content
- YouTube comments via `youtube-comment-downloader`
- Colloquial Indonesian lexicon from nasalsabila/`kamus-alay`
