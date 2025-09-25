
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
