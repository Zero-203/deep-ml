import torch

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics using PyTorch.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    log_probs_t = torch.as_tensor(log_probs)
    correct_answers_t = torch.as_tensor(correct_answers)
    rounds_n, choices_n = log_probs_t.shape
    predictions = torch.argmax(log_probs_t,dim=1)
    correct_n = 0
    correct_prob = 0.0
    for idx in range(rounds_n):
        if predictions[idx] == correct_answers_t[idx]:
            correct_n += 1 
        correct_prob += torch.exp(log_probs_t[idx][correct_answers_t[idx]])/(torch.sum(torch.exp(log_probs_t[idx])))
    
    accuracy = correct_n / rounds_n
    avg_correct_prob = correct_prob / rounds_n

    return {
        'accuracy':round(accuracy,4),
        'predictions':predictions.tolist(),
        'avg_correct_prob':avg_correct_prob
    }


