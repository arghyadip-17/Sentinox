# 🧠 Sentinox — AI Sentiment Analyzer

🚀 **Sentinox AI** is a powerful NLP-based web application that analyzes the sentiment of text using state-of-the-art transformer models. It classifies text into **Positive, Neutral, or Negative** with probability scores.

---

## 🌐 Live Demo

📍 https://sentinox.onrender.com
- It may take times due to large models size.

---

## ✨ Features

* 🔍 **Single Text Analysis** — Instantly analyze sentiment of any sentence
* 📊 **Batch Processing** — Analyze multiple lines of text at once
* 📁 **CSV Upload Support** — Upload datasets and get predictions
* 📈 **Probability Scores** — Get confidence for each sentiment class
* ⚡ **Fast & Efficient** — Uses cached model loading
* 🎨 **Clean UI** — Full-width layout with hidden sidebar

---

## 🧠 Model Used

* 🤖 `cardiffnlp/twitter-roberta-base-sentiment`
* 3-class classification:

  * NEGATIVE
  * NEUTRAL
  * POSITIVE

---

## 🛠️ Tech Stack

* **Frontend & Backend:** Streamlit
* **Machine Learning:** Transformers (HuggingFace)
* **Deep Learning:** PyTorch
* **Data Handling:** Pandas, NumPy

---

## 📂 Project Structure

```
Sentinox-AI/
│── app.py
│── requirements.txt
│── README.md
```

---
## ⚙️ Installation & Run Locally

1️⃣ Clone Repository
```
git clone https://github.com/arghyadip-17/sentinox.git
cd sentinox
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Run App
```
streamlit run app.py
```
---

## 📸 Features Preview

* 🔹 Real-time sentiment prediction
* 🔹 Interactive charts for batch analysis
* 🔹 Downloadable CSV results

---

## 👨‍💻 Author

**Arghyadip Ghosh**

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Share it

---

### 💬 Tagline

> *“Decode Emotions. Instantly.”*

---
