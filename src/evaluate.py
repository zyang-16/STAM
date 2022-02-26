import itertools
import numpy as np
import sys
import heapq
from concurrent.futures import ThreadPoolExecutor

def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def ndcg_at_k(rank, ground_truth, K):
    rank = rank[:K]
    user_hits = np.array([1 if i in ground_truth else 0 for i in rank], dtype=np.float32)
    if len(ground_truth) >= K:
        ideal_rels = np.ones(K)
    else:
        ideal_rels = np.pad(np.ones(len(ground_truth)), (0, K-len(ground_truth)), 'constant')
    dcg = np.sum(user_hits / np.log2(np.arange(2, len(user_hits)+2)))
    idcg = np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels)+2))) 
    ndcg = dcg / idcg
    return ndcg


def mrr_at_k(rank, ground_truth, K):
    rank = rank[:K] 
    rr_list = []
    for item in ground_truth:
        if item not in rank:
            rr = 0.
        else:
            r = rank.index(item) + 1
            rr = 1.0 / r
        rr_list.append(rr)
    mrr = sum(rr_list)/len(rr_list)
    return mrr


def hit_at_k(rank, ground_truth, K):
    rank = rank[:K]  
    hits = [1 if item in ground_truth else 0 for item in rank]
    if np.sum(np.array(hits)) > 0:
        return 1.
    else:
        return 0.

def evaluate(score_matrix, test_items, Ks=[20, 50], thread_num=10):
    def _eval_one_user(idx):
        ranking = score_matrix[idx]  # all scores of the test user
        test_item = test_items[idx]  # all test items of the test user
        top_k = max(Ks)

        ranking = [p for p in ranking if p!=-1][:top_k]   # Top-K items 
        mrr, ndcg, hits = [], [], [] 
        for K in Ks: 
            mrr.append(mrr_at_k(ranking, test_item, K))
            ndcg.append(ndcg_at_k(ranking, test_item, K)) 
            hits.append(hit_at_k(ranking, test_item, K)) 
        return {'mrr': np.array(mrr), 'ndcg': np.array(ndcg), 'hits': np.array(hits)} 

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_user, range(len(test_items)))

    result = list(batch_result)  # generator to list
    return np.array(result)   # list to ndarray

















