# ⚙️ Industrial Mechanical AI Assistant

An AI-powered predictive maintenance system that combines Machine Learning + RAG-based LLMs to assist industrial operators in real-time decision-making.

---

## 🚀 Problem Statement
Industrial machines often fail without early warning, leading to costly downtime and maintenance issues. Traditional monitoring systems lack intelligent reasoning and explanation capabilities.

---

## 💡 Solution
This system combines:
- Machine Learning for failure prediction
- RAG (Retrieval Augmented Generation) for intelligent explanations
- LLM-based assistant for natural language interaction

It helps operators understand machine health and take preventive actions.

---

## 🧠 Tech Stack
- Python  
- Streamlit  
- Scikit-learn (RandomForest)  
- LangChain  
- FAISS Vector Database  
- Hugging Face Embeddings  
- Groq LLaMA 3  
- RAG Pipeline  

---

## ⚙️ Features
- Real-time sensor monitoring  
- Machine failure prediction using ML  
- AI-powered explanations using RAG  
- Wikipedia-based industrial knowledge retrieval  
- Natural language Q&A assistant  

---

## 🔄 System Workflow
1. Sensor data input  
2. ML model predicts failure probability  
3. RAG retrieves relevant knowledge  
4. LLM generates explanation + suggestions  
5. Output displayed in Streamlit dashboard  

---

## 📷 Screenshot

![Industrial AI Dashboard](./dashboard.png)

*Real-time Industrial AI dashboard showing ML prediction, sensor thresholds, and RAG-based assistant.*

---

## 🌐 Live Demo

This project is deployed on Hugging Face Spaces.

👉 https://huggingface.co/spaces/DipsPtl/Industrial-Mech-AI-Assistant

---

## 💻 Run Locally (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
