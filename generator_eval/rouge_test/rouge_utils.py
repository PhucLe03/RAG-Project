from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaLLM
# from utils import *
from rouge_score import rouge_scorer

import os
import json

template = """
You are a helpful and friendly Next.js assistant. 
Your responsibility is to answer user queries about Next.js. 
Answer the question based only and only on the given context below (which got from Next.js documentation). If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

parser = StrOutputParser()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

class ROUGE:
    def __init__(self):
        pass
    def set_testcases(self, testcases):
        self.testcases = testcases
    def set_model(self, model):
        self.model = OllamaLLM(model=model)
        self.chain = prompt | self.model | parser
        self.output_file = os.path.join("results", f"{model.replace(':', '-')}.json")
    def run_tests(self):
        results = []
        for test_case in self.testcases:
            answer = self.chain.invoke({"context": test_case["context"], "question": test_case["question"]})
            results.append({
                "id": test_case["id"],
                "answer": answer
            })
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
    def load_results(self, output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            self.results = json.load(f)
            
    def merge_results(self, output_file, test_cases_file):
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Create a dictionary from results using 'id' for quick lookup
        results_dict = {item["id"]: item for item in results}

        # Merge the test cases with corresponding results
        self.merged_data = []
        for test_case in test_cases:
            case_id = test_case["id"]
            if case_id in results_dict:
                self.merged_data.append({
                    "id": case_id,
                    "answer": results_dict[case_id]["answer"],  # Get the answer from results
                    "reference_summary": test_case["reference_summary"]  # Keep reference summary from test cases
                })
    
    def score_rouge(self):
        rouge1_precision = []
        rouge1_recall = []
        rouge1_fmeasure = []
        rouge2_precision = []
        rouge2_recall = []
        rouge2_fmeasure = []
        rouge3_precision = []
        rouge3_recall = []
        rouge3_fmeasure = []
        rougeL_precision = []
        rougeL_recall = []
        rougeL_fmeasure = []
        for item in self.merged_data:
            score = scorer.score(item["answer"], item["reference_summary"])
            rouge1 = score['rouge1']
            rouge2 = score['rouge2']
            rouge3 = score['rouge3']
            rougeL = score['rougeL']
            rouge1_precision.append(rouge1[0])
            rouge1_recall.append(rouge1[1])
            rouge1_fmeasure.append(rouge1[2])
            rouge2_precision.append(rouge2[0])
            rouge2_recall.append(rouge2[1])
            rouge2_fmeasure.append(rouge2[2])
            rouge3_precision.append(rouge3[0])
            rouge3_recall.append(rouge3[1])
            rouge3_fmeasure.append(rouge3[2])
            rougeL_precision.append(rougeL[0])
            rougeL_recall.append(rougeL[1])
            rougeL_fmeasure.append(rougeL[2])
        # return avg
        avg_rouge1_precision = sum(rouge1_precision) / len(rouge1_precision)
        avg_rouge1_recall = sum(rouge1_recall) / len(rouge1_recall)
        avg_rouge1_fmeasure = sum(rouge1_fmeasure) / len(rouge1_fmeasure)
        avg_rouge2_precision = sum(rouge2_precision) / len(rouge2_precision)
        avg_rouge2_recall = sum(rouge2_recall) / len(rouge2_recall)
        avg_rouge2_fmeasure = sum(rouge2_fmeasure) / len(rouge2_fmeasure)
        avg_rouge3_precision = sum(rouge3_precision) / len(rouge3_precision)
        avg_rouge3_recall = sum(rouge3_recall) / len(rouge3_recall)
        avg_rouge3_fmeasure = sum(rouge3_fmeasure) / len(rouge3_fmeasure)
        avg_rougeL_precision = sum(rougeL_precision) / len(rougeL_precision)
        avg_rougeL_recall = sum(rougeL_recall) / len(rougeL_recall)
        avg_rougeL_fmeasure = sum(rougeL_fmeasure) / len(rougeL_fmeasure)

        # scores = {
        #     "rouge1_precision": avg_rouge1_precision,
        #     "rouge1_recall": avg_rouge1_recall,
        #     "rouge1_fmeasure": avg_rouge1_fmeasure,
        #     "rougeL_precision": avg_rougeL_precision,    
        #     "rougeL_recall": avg_rougeL_recall,
        #     "rougeL_fmeasure": avg_rougeL_fmeasure
        # }
        # nested
        scores = {
            "rouge1": {
                "precision": avg_rouge1_precision,
                "recall": avg_rouge1_recall,
                "fmeasure": avg_rouge1_fmeasure
            },
            "rouge2": {
                "precision": avg_rouge2_precision,
                "recall": avg_rouge2_recall,
                "fmeasure": avg_rouge2_fmeasure
            },
            "rouge3": {
                "precision": avg_rouge3_precision,
                "recall": avg_rouge3_recall,
                "fmeasure": avg_rouge3_fmeasure
            },
            "rougeL": {
                "precision": avg_rougeL_precision,
                "recall": avg_rougeL_recall,
                "fmeasure": avg_rougeL_fmeasure
            }
        }
        
        return scores