from sklearn.covariance import MinCovDet
from rouge_score import rouge_scorer
from sentence_transformers import util
import heapq
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL



def getSentenceSimilarity(generations, answers, SenSimModel):
    gen_embeddings = SenSimModel.encode(generations)
    ans_embeddings = SenSimModel.encode(answers)
    similarity = util.cos_sim(gen_embeddings, ans_embeddings)
    return similarity.item()


#return activation map for last token
def get_activation_map(activations,num_tokens=None):
    activation_map=[]
    if num_tokens:
        for i in range(len(num_tokens)):
            activation_map.append(activations[num_tokens[i]-3][:,i,:])
        
    else:
        activation_map=activations[-1].transpose(1,0,2)[0]

    return np.asarray(activation_map,dtype=float)


def jaccard_similarity_above_zero(rows1, rows2):
    set1 = set(np.where(rows1.flatten() > 0)[0])
    set2 = set(np.where(rows2.flatten() > 0)[0])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def compute_matrix_similarities(activation_maps,alpha=0.5,beta=0.5):
    n = len(activation_maps)
    total_cosine_similarity = 0
    total_jaccard_similarity = 0
    count = n * (n - 1) / 2 
    result=0
    
    for i in range(n - 1):
        for j in range(i + 1, n):

            row_similarities = [
                cosine_similarity([row1], [row2])[0, 0] 
                for row1, row2 in zip(activation_maps[i], activation_maps[j])
            ]


            matrix1, matrix2 = activation_maps[i], activation_maps[j]
            jaccard_similarity = np.mean([
                jaccard_similarity_above_zero(matrix1[k:k+2], matrix2[k:k+2])
                for k in range(matrix1.shape[0] - 1)
            ])

            result+=alpha*np.mean(row_similarities)+beta*jaccard_similarity
    
    return result/count

