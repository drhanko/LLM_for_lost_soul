# Final-project-in-analyse-system
# NLP Emotion Classification Pipeline

This project provides an end-to-end pipeline for emotion classification, including:

- Data collection from multiple datasets
- Data preprocessing and merging
- Model training
- Model evaluation
- Optional dataset loading from Google Drive

---

## 📁 Project Structure

.
├── main.py
├── data_collection.py
├── data_preprocessing.py
├── data_process_ready.py
├── predict_model.py
├── predict_evaluation.py
├── config.py
├── data/
├── results/
└── .env

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If using Google Drive:

```bash
pip install gdown python-dotenv
```

---

### 2. Environment Variables (Optional)

Create a `.env` file:

```env
GOOGLE_DRIVE_FOLDER_ID="Will assign by the owner"
```

---
default dataset_link = ["dailydialog","empathetics","go_emotions","mixed_data"]

## 🚀 Usage

### 🔹 1. Pre-train (Full data pipeline)

Run data collection, preprocessing, and dataset preparation:

python main.py pre_train

### 🔹 2. Train

Train the model using prepared dataset:

python main.py train dataset_link<dataset_link>


---

### 🔹 3. Evaluation

Evaluate the trained model.

#### ✔ Local model

python main.py eval --switch local


#### ✔ Google Drive model / dataset
python main.py eval --switch google --dataset_link <dataset_link>


---

## 📦 Dataset

The project supports:

- EmpatheticDialogues
- DailyDialog
- GoEmotions
- Mixed

Datasets are processed into:

data/<dataset_name>/split/
- train.jsonl
- validation.jsonl
- test.jsonl

---

## ☁️ Google Drive Integration

If using Google Drive:

- Ask owner for further use
- The script will download and cache locally


## 🧠 Model

- Training: `predict_model.py`
- Evaluation: `predict_evaluation.py`
- Models are saved in:

results/

---

## ⚠️ Notes

- Google Drive links must be public ("Anyone with the link")
- Large datasets will be downloaded locally
- Ensure `.env` is properly loaded if used

---

## 🛠️ Future Improvements

- Add logging system
- Support multiple model versions
- Add experiment tracking (e.g., MLflow)
- Improve dataset caching

---

## 📬 Contact

Feel free to open an issue for questions or improvements.

## AI portion
1.Readme(protion modified)
2.torch setting (protion modified)
3 tokenize setting (protion modified)