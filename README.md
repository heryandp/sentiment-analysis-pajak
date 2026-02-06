# Sentiment Analysis Pajak

Indonesian tax sentiment analysis experiments and notebooks. This repo contains notebooks, scripts, and plots used to collect data, preprocess text, and train/evaluate sentiment models.

## Contents
- Notebooks for data collection, preprocessing, training, and evaluation
- A utility script for JSON to CSV conversion
- Example outputs and comparison plots

## Data and Models
Large datasets and model artifacts are excluded from git. See `.gitignore` for excluded paths. If you need to reproduce results, place your data in the same directory structure used by the notebooks.

## Quick Start
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
```
Open the notebooks in Jupyter and run cells in order.

## Notes
- Notebooks are the main source of truth.
- Update `requirements.txt` if you add new dependencies.
