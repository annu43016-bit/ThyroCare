THYROCARE AI — REAL-WORLD DATA PROJECT
========================================

This project uses a public thyroid dataset mirror attributed to the UCI Machine
Learning Repository. It is historical, anonymized research data, not live or
recent patient data. The dataset is sourced from a clinical dataset with
diagnosis labels. The application converts the original diagnosis field into
a binary target for a classroom demonstration:
  0 = diagnosis marked "negative"
  1 = any other diagnosis label

DATA SOURCE
- UCI Machine Learning Repository, Thyroid Disease:
  https://archive.ics.uci.edu/dataset/102/thyroid%2Bdisease
- Public CSV mirror used by the automatic downloader:
  https://github.com/Poojitha369/DSA6000/blob/main/Thyroid_diagnosis.csv
- Raw CSV URL:
  https://raw.githubusercontent.com/Poojitha369/DSA6000/main/Thyroid_diagnosis.csv

The app automatically downloads the CSV into data/Thyroid_diagnosis.csv the
first time it starts. An internet connection is needed for that first run.
After downloading, the file is kept locally and later runs can work offline.

HOW TO RUN ON WINDOWS
1. Extract this ZIP file.
2. Open the extracted project folder.
3. Click the folder address bar, type cmd, and press Enter.
4. Run these commands one at a time:
     python -m pip install -r requirements.txt
     python -m pip install reportlab
     python app.py
5. Open http://127.0.0.1:5000 in your browser.

If "python" is not recognized, try using "py" instead of "python".

PROJECT NOTES
- Uses four models: Decision Tree, KNN, SVM, and Logistic Regression.
- Includes model metrics, class distribution, confusion matrix, ROC curve,
  feature-importance chart, prediction form, CSV export and PDF export.
- Target labels are simplified for a college demo; this is not a clinically
  validated diagnosis model.
- Historical, anonymized research records should not be interpreted as current
  medical guidance. Never use the app to diagnose or treat a person.
- Charts load Chart.js from a CDN and therefore need internet access to display.
- Please cite Ross Quinlan (1986), Thyroid Disease, UCI Machine Learning
  Repository, DOI: https://doi.org/10.24432/C5D010 when describing the dataset.
