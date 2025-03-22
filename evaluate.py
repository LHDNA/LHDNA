import os
import numpy as np
import pickle as pkl
import evaluate
from rouge_score import rouge_scorer
import math
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from metric import *
# from plot import *
import time
import csv
import os 
USE_Roberta = False
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
SenSimModel = SentenceTransformer('sentence-transformers/nli-roberta-large')

def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()

def getAcc(resultDict, file_name):
    correctCount = 0
    correct_index=[]
    index=0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if "coqa" in file_name or "TruthfulQA" in file_name:
            additional_answers = item["additional_answers"]
            rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores))
        if rougeScore>0.7:
            correctCount += 1
            correct_index.append(index)
        index=index+1
    print("Acc:", 1.0*correctCount/len(resultDict))
    return correct_index



def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]


def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    LHDNA = []
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        LHDNA.append(item["lhdna"])

        if USE_Roberta:
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ansGT, SenSimModel) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity>0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
            pass
        else:
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            if rougeScore>0.7:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)

    fpr, tpr, thresholds = roc_curve(Label, LHDNA)
    AUROC = auc(fpr, tpr)
    print("AUROC-LHDNA:", AUROC)

    rho_LHDNA = getPCC(Score, LHDNA)
    print("rho_LHDNA:", rho_LHDNA)


def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt



def getTruthfulQAAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)



def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


if __name__ == "__main__":
    file_name="results/opt-1.3b_coqa/0.pkl"
    f = open(file_name, "rb")
    resultDict = np.asarray(pkl.load(f))
    getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)
