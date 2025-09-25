import os
import json
import time
import pandas as pd
import re
from dotenv import load_dotenv
from typing import List, Dict

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document, BaseRetriever

# FlashRank
from flashrank import Ranker, RerankRequest
from pydantic import Field

# Metrics
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

MAX_TOKENS = 512
TOP_K = 20
TOP_N = 5

MODELS_TO_TEST = {
    "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "DistilBERT_Arabic": "asafaya/bert-base-arabic",
    "AraBERT_Large": "aubmindlab/bert-large-arabertv02"
    }

RERANKER_CONFIGURATIONS = {
    "no_reranking": {"use_reranking": False},
    "flashrank_arabic": {"use_reranking": True, "method": "flashrank"}
}

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
FIQH_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي بالاعتماد على السياق المعطى.
أعد صياغة الإجابة بشكل مختصر مع تحديد الخيار الصحيح (أ، ب، ج).

السياق:
{context}

السؤال: {question}

الخيارات:
{options}

"""

# --------------------------------------------------
# UTILITAIRES
# --------------------------------------------------
def load_flashrank():
    return Ranker(model_name="miniReranker_arabic_v1")

def validate_answer(model_answer: str, correct_answer: str) -> bool:
    """Méthode d'extraction de réponse améliorée"""
    if not model_answer or not correct_answer:
        return False
    
    # Mapping bidirectionnel des réponses arabes ↔ latines
    arabic_to_latin = {'أ': 'A', 'ب': 'B', 'ج': 'C'}
    
    # Normalisation robuste de la réponse correcte
    correct_normalized = correct_answer.strip().upper()
    if correct_normalized in arabic_to_latin:
        correct_normalized = arabic_to_latin[correct_normalized]
    
    # Pattern principal : recherche avec délimiteurs contextuels
    primary_pattern = r'[ABCأبج](?=[.\s:،؛]|$)'
    matches = re.findall(primary_pattern, model_answer)
    
    # Pattern de fallback : mots complets uniquement
    if not matches:
        fallback_pattern = r'\b[ABCأبج]\b'
        matches = re.findall(fallback_pattern, model_answer)
    
    # Pattern de secours : toute occurrence
    if not matches:
        emergency_pattern = r'[ABCأبج]'
        matches = re.findall(emergency_pattern, model_answer)
    
    if not matches:
        return False
    
    # Prendre la dernière occurrence trouvée
    last_answer = matches[-1].upper()
    
    # Conversion vers format latin pour comparaison uniforme
    if last_answer in arabic_to_latin:
        last_answer = arabic_to_latin[last_answer]
    
    return last_answer == correct_normalized

# --------------------------------------------------
# FLASHRANK RETRIEVER
# --------------------------------------------------
class FlashRankRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    flashranker: Ranker = Field(...)
    top_n: int = Field(default=5)

    def get_relevant_documents(self, query, **kwargs):
        try:
            docs = self.base_retriever.get_relevant_documents(query)
            passages = [{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = self.flashranker.rerank(rerank_request)
            return [docs[int(r['id'])] for r in reranked[:self.top_n]]
        except Exception as e:
            print(f"⚠️ FlashRankRetriever error: {e}")
            return []

    async def aget_relevant_documents(self, query, **kwargs):
        return self.get_relevant_documents(query, **kwargs)

# --------------------------------------------------
# CLASSE PRINCIPALE
# --------------------------------------------------
class SimpleRAGComparator:
    def __init__(self, qcm_data: Dict, text_file_path: str):
        self.qcm_data = qcm_data
        self.text_file_path = text_file_path

    def initialize_rag_system(self, model_name: str, model_path: str, config: Dict):
        print(f"Initialisation: {model_name} | Re-ranking: {config['use_reranking']}")

        with open(self.text_file_path, 'r', encoding='utf-8') as f:
            raw_chunks = f.read().split('***')
        texts = [Document(page_content=chunk.strip()) for chunk in raw_chunks if chunk.strip()]

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )

        vectorstore = FAISS.from_documents(texts, embeddings)
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

        retriever = base_retriever
        if config["use_reranking"] and config.get("method") == "flashrank":
            flashranker = load_flashrank()
            retriever = FlashRankRetriever(base_retriever=base_retriever, flashranker=flashranker, top_n=TOP_N)
            print("✅ FlashRank Arabic re-ranker initialized")

        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                        temperature=0.05, top_k=40, top_p=0.95, num_predict=MAX_TOKENS)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=FIQH_TEMPLATE, input_variables=["context","question","options"])},
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    def run_comparison(self, question_ids: List[str]):
        results = []
        total_start_time = time.time()

        total_combinations = len(MODELS_TO_TEST) * len(RERANKER_CONFIGURATIONS)
        print(f"Total combinations to test: {total_combinations}")

        for model_name, model_path in MODELS_TO_TEST.items():
            for config_name, config_params in RERANKER_CONFIGURATIONS.items():
                print(f"\n🔎 Testing {model_name} + {config_name}")
                try:
                    qa_chain = self.initialize_rag_system(model_name, model_path, config_params)
                except Exception as e:
                    print(f"❌ Init error {model_name}+{config_name}: {e}")
                    results.append({
                        "model": model_name,
                        "configuration": config_name,
                        "error_message": str(e)
                    })
                    continue

                config_results = []
                for qid in question_ids:
                    q = self.qcm_data[qid]
                    question = q["question"]
                    options = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
                    correct_answer = q["answer_letter"]

                    try:
                        start_time = time.time()
                        response = qa_chain({"question": question, "options": options, "chat_history": []})
                        response_time = time.time() - start_time

                        model_answer = response.get("answer", "")
                        is_correct = validate_answer(model_answer, correct_answer)

                        smoothie = SmoothingFunction().method4
                        bleu = sentence_bleu([[correct_answer]], model_answer, smoothing_function=smoothie)
                        f1 = f1_score([1], [1 if is_correct else 0], zero_division=0)

                        entry = {
                            "model": model_name,
                            "configuration": config_name,
                            "question_id": qid,
                            "question": question,
                            "correct_answer": correct_answer,
                            "model_answer": model_answer,
                            "is_correct": is_correct,
                            "bleu": bleu,
                            "f1": f1,
                            "response_time": response_time,
                            "error_message": None
                        }
                        results.append(entry)
                        config_results.append(entry)

                    except Exception as e:
                        print(f"⚠️ Error {model_name}+{config_name} on Q{qid}: {e}")
                        results.append({
                            "model": model_name,
                            "configuration": config_name,
                            "question_id": qid,
                            "error_message": str(e)
                        })
                        continue

                # Résumé intermédiaire
                if config_results:
                    df_config = pd.DataFrame(config_results)
                    acc = df_config["is_correct"].mean()
                    mean_time = df_config["response_time"].mean()
                    mean_bleu = df_config["bleu"].mean()
                    mean_f1 = df_config["f1"].mean()
                    print(f"\n📊 Résumé {model_name} + {config_name}:")
                    print(f"   Accuracy       : {acc:.2%}")
                    print(f"   Mean Resp. Time: {mean_time:.2f} sec")
                    print(f"   Mean BLEU      : {mean_bleu:.3f}")
                    print(f"   Mean F1        : {mean_f1:.3f}")

        df = pd.DataFrame(results)
        df.to_csv("rag_comparison_flashrank_arabic.csv", index=False, encoding="utf-8-sig")

        print("\n=== FINAL SUMMARY ===")
        if not df.empty:
            print(df.groupby(["model","configuration"]).agg({
                "is_correct":"mean",
                "bleu":"mean",
                "f1":"mean",
                "response_time":"mean"
            }))

        print(f"\nTotal execution time: {(time.time()-total_start_time)/60:.2f} minutes")
        return df

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    qcm_file = "qcm_test_140QA.json"
    text_file = "dataset_700QA.txt"

    if not os.path.exists(qcm_file):
        print(f"❌ File not found: {qcm_file}")
        return

    with open(qcm_file, 'r', encoding='utf-8') as f:
        qcm_data = json.load(f)

    comparator = SimpleRAGComparator(qcm_data, text_file)
    question_ids = list(qcm_data.keys())
    results = comparator.run_comparison(question_ids)

if __name__ == "__main__":
    main()