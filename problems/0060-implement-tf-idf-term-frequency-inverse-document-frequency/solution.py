import torch
from typing import List
import math
from math import log

def compute_tf_idf(corpus: List[List[str]], query: List[str]) -> torch.Tensor:
    """
    Compute TF-IDF scores for a query against a corpus of documents using PyTorch.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: torch.Tensor of shape (num_docs, num_query_words) with TF-IDF scores
             rounded to five decimal places
    """
    num_docs = len(corpus)
    num_queries = len(query)
    res = torch.zeros([num_docs,num_queries])
    for i in range(num_queries):
        cnt_show, cnt_doc = 0, 0
        for j in range(num_docs):
            cnt_doc = 0
            for word in corpus[j]:
                cnt_doc += 1 if word == query[i] else 0
            res[j][i] = cnt_doc / len(corpus[j])
            cnt_show += 1 if cnt_doc else 0
        idf = log(num_docs + 1) - log(cnt_show + 1) + 1
        for j in range(num_docs):
            res[j][i] *= idf

    return res



