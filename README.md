# Eksperimen_SML_Gregorius-Willson

Eksperimen preprocessing dataset **Telco Customer Churn** untuk submission Dicoding MSML (Advanced).

## Struktur

```
telco_raw/
preprocessing/
  Eksperimen_Gregorius-Willson.ipynb
  automate_Gregorius-Willson.py
  telco_preprocessing/
.github/workflows/preprocess.yml
```

## Menjalankan lokal

```bash
pip install pandas scikit-learn jupyter
python preprocessing/automate_Gregorius-Willson.py
```

Workflow GitHub Actions menjalankan preprocessing otomatis pada setiap push ke `main`.
