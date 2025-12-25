# 🧠 Aspect-Based Customer Feedback Analysis Dashboard

A simple **Streamlit** application for analyzing customer feedback by aspect using deep learning.  
The system applies **Aspect-Based Sentiment Analysis (ABSA)** to extract aspects, determine sentiment, visualize result distributions, and provide recommendations for restaurants or laptop stores.

---

## 📌 Project Overview

This project focuses on building an AI-based system that analyzes customer reviews at the **aspect level** (e.g., food quality, service, price, laptop performance, design, etc.).  
The model is trained on the **SemEval 2016 Task 5 dataset** and uses **distilroberta-base** for efficient and accurate sentiment analysis.

**Key Features:**

- Analyze a single sentence or multiple sentences from a text file  
- Aspect-level sentiment prediction  
- Result distribution visualization  
- Recommendation generation based on analysis results  

---

## 🛠️ Technologies Used

- Python  
- PyTorch  
- Hugging Face Transformers  
- distilroberta-base  
- Streamlit  
- SemEval 2016 Task 5 Dataset  

---

## 📊 Dataset

- **Laptop domain:** 2,301 sentences  
- **Restaurant domain:** 1,907 sentences  
- Data cleaned by removing missing labels and rare aspect categories  
- Final dataset split: 80% training – 20% testing  

**Note:** The official test set of SemEval 2016 Task 5 is not publicly available, so evaluation is performed on a self-split test set.

---

## 🚀 How to Run the App Locally

1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
2️⃣ Run the Streamlit app:
```bash
streamlit run streamlit_app.py


3️⃣ Open in browser:
Streamlit will automatically open the application in your default browser.

📈 Evaluation Metrics

F1-score (main metric)

Accuracy (supporting metric)

F1-score is used as the primary metric due to class imbalance in aspect categories.

