# 📊 RAG Benchmark with Arabic Embeddings, FlashRank & Ollama

This project benchmarks **three Arabic embedding models** in a **RAG (Retrieval-Augmented Generation)** pipeline, applied to **Maliki Fiqh datasets**.  
It compares the performance **with and without FlashRank re-ranking**, using **Ollama LLM (llama3:8b)**.

---

## 🚀 Features
- ✅ Compare **3 Arabic embeddings** (CAMeLBERT, DistilBERT, AraBERT Large)  
- ✅ Test with **and without FlashRank Arabic reranker**  
- ✅ Dataset: `dataset_700QA.txt` (700 QA in Maliki fiqh)  
- ✅ Evaluation on `qcm_test_140QA.json` (140 MCQ questions with correct answers)  
- ✅ Metrics: **Accuracy, BLEU score, F1-score, Response time**  
- ✅ Results exported to CSV  

---

## 📂 Repository Structure
comparaison-RAG-arabe/
│── comparaison_3modeles_embeddings_flasreranking_ollama.py # main script

│── requirements.txt # dependencies

│── README.md # project overview

│── GUIDE.md # detailed usage guide

│── data/

│ ├── qcm_test_140QA.json # test QCM file

│ └── dataset_700QA.txt # knowledge dataset

│── results/

│ └── rag_comparison_flashrank_arabic.csv # results (generated)
▶️ Usage

Make sure Ollama is running locally with the llama3:8b model:

ollama run llama3:8b


Then run the benchmark:

python comparaison_3modeles_embeddings_flasreranking_ollama.py

📊 Output
The script performs 6 experiments:

CAMeLBERT + no reranking

CAMeLBERT + FlashRank

DistilBERT + no reranking

DistilBERT + FlashRank

AraBERT Large + no reranking

AraBERT Large + FlashRank

Generated results:

results/rag_comparison_flashrank_arabic.csv

Intermediate summaries in terminal (accuracy, BLEU, F1, response time)

Final aggregated performance table

