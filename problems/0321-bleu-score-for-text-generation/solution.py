import torch
import math
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences
    using PyTorch tensor operations for numerical computations.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1 as a Python float
    """
    # Your code here
    p_list = []
    c = len(candidate)
    if c==0:
        return 0.0

    for n in range(1,max_n+1):
        can_dic={}
        
        ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) -
        n + 1)]
        for token in ngrams:
            if can_dic.get(token) is None:
                can_dic[token] = 1
            else:
                can_dic[token] += 1

        total_hit = 0
        for item in can_dic.items():
            key,val=item
            max_hit = 0
            
            for ref in references:
                hit = 0
                ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) -
        n + 1)]
                for token in ref_ngrams:
                    hit += 1 if token == key else 0
                max_hit = max(max_hit,hit)
            total_hit += min(val,max_hit)
        
        p_list.append(total_hit/(c-n+1))
    
    close_r = len(references[0])
    for ref in references:
        if abs(len(ref)-c)<= abs(close_r - c) and len(ref)<close_r:
            close_r = len(ref)

    bp = 1 if c >= close_r else math.exp(1-close_r/c)

    p_t = torch.as_tensor(p_list)

    zero = False
    for p in p_list:
        if p==0:
            zero=True
            break
    
    if zero:
        bleu = 0.0
    else:
        bleu = bp * torch.exp(torch.sum(1/max_n*torch.log(p_t))).item()
    return round(bleu,4)    