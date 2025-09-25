
---

## 📄 Fichier 3 : `GUIDE.md`
```markdown
# 📖 Detailed Guide – RAG Benchmark with Arabic Embeddings

This guide explains in detail how the script works, the evaluation methodology, and how to extend the benchmark.

---

## 1️⃣ Project Goal
The goal is to benchmark **Arabic embeddings** in a **RAG pipeline** for answering **Maliki Fiqh questions**.  
We evaluate whether **FlashRank Arabic reranker** improves retrieval quality and final answers.

---

## 2️⃣ Workflow Overview
1. **Dataset Preparation**
   - Knowledge base: `dataset_700QA.txt` (700 QA pairs, separated by `***`)
   - Test set: `qcm_test_140QA.json` (140 multiple-choice questions with correct answers)

2. **Embeddings**
   - CAMeLBERT  
   - DistilBERT Arabic  
   - AraBERT Large  

3. **Retriever**
   - FAISS similarity search (`k=20`)  
   - Optionally enhanced with **FlashRank Arabic reranker** (`top_n=5`)

4. **LLM**
   - Ollama `llama3:8b`  
   - Used as the generator to answer based on retrieved context

5. **Evaluation**
   - Each answer is compared with the gold label using:
     - **Accuracy** (correct vs incorrect)
     - **BLEU** (text similarity)
     - **F1-score** (binary classification metric)
     - **Response time**

---

## 3️⃣ Running the Script
```bash
python comparaison_3modeles_embeddings_flashreranking_ollama.py

The script will:

Load datasets (qcm_test_140QA.json, dataset_700QA.txt)

Run 6 experiments (3 embeddings × 2 configs)

Print intermediate summaries

Export results into results/rag_comparison_flashrank_arabic.csv

5️⃣ How to Extend
🔧 Add new embeddings: update MODELS_TO_TEST in the script

🔧 Change dataset: replace dataset_700QA.txt with your corpus

🔧 Modify LLM: change OLLAMA_MODEL (e.g., gemma2:9b)

🔧 Custom reranking: plug in another reranker in FlashRankRetriever

6️⃣ Limitations
Requires Ollama running locally

Evaluation based mainly on multiple-choice questions

BLEU may not fully capture semantic correctness

7️⃣ Conclusion
This benchmark provides a reproducible framework for testing Arabic embeddings in RAG pipelines, especially for Islamic jurisprudence datasets.
It shows the impact of FlashRank Arabic on retrieval quality and final accuracy.
