def detect_contamination(test_examples, train_corpus, n, threshold):
    """
    Args:
        test_examples: list[str] - test set documents
        train_corpus:  list[str] - training set documents
        n: int - n-gram size
        threshold: float - coverage ratio in [0, 1] above which an example is flagged

    Returns:
        float - contamination percentage in [0, 100]
    """
    tokenized_test = []
    tokenized_train = []
    for example in test_examples:
        tokenized_test.append(example.split(" "))
    for example in train_corpus:
        tokenized_train.append(example.split(" "))

    train_ngram=set()
    for train in tokenized_train:
        for idx in range(len(train)-n):
            train_ngram.add(tuple(train[idx:idx+n]))
    
    covercnt, Ntest = 0, len(tokenized_test)
    for test in tokenized_test:
        covered = [False]*len(test)
        for idx in range(len(test)-n):
            if tuple(test[idx:idx+n]) in train_ngram:
                covered[idx:idx+n-1] = [True] * n
        rate = sum(covered) / len(test)
        if rate >= threshold:
            covercnt += 1
    
    return covercnt / Ntest * 100.0