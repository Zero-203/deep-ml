import torch
import math

def isolation_forest(X: torch.Tensor, n_trees: int, sample_size: int, random_state: int = 42) -> torch.Tensor:
    """
    Implements the Isolation Forest anomaly detection algorithm using PyTorch.

    Parameters:
    - X: 2D torch.Tensor of shape (n_samples, n_features)
    - n_trees: Number of isolation trees to build
    - sample_size: Number of samples used to build each tree (subsampling without replacement)
    - random_state: Seed for reproducibility

    Returns:
    - scores: 1D torch.Tensor of anomaly scores. Higher values (close to 1) indicate anomalies,
              values around 0.5 are normal.
    """
    torch.manual_seed(random_state)

    n_samples, n_features = X.shape
    # Ensure sample_size does not exceed the number of available samples
    actual_sample_size = min(sample_size, n_samples)
    max_depth = math.ceil(math.log2(actual_sample_size))

    # Helper to compute the average path length of an unsuccessful search in a BST
    def c(n: int) -> float:
        if n <= 1:
            return 0.0
        # Harmonic number H(n-1) approximated using the Euler-Mascheroni constant
        H = math.log(n - 1) + 0.5772156649
        return 2.0 * H - 2.0 * (n - 1) / n

    # Internal representation of a tree node
    class Node:
        def __init__(self, size):
            self.is_leaf = True
            self.size = size
            self.split_feat = None
            self.split_val = None
            self.left = None
            self.right = None

    def build_tree(data: torch.Tensor, depth: int) -> Node:
        """Recursively builds an isolation tree from data."""
        n = data.shape[0]
        # Stop splitting if max depth reached or node has at most 1 sample
        if depth >= max_depth or n <= 1:
            return Node(n)

        # Randomly choose a feature
        feat = torch.randint(0, n_features, (1,)).item()
        min_val = data[:, feat].min().item()
        max_val = data[:, feat].max().item()

        # If all values are identical, cannot split further
        if min_val == max_val:
            return Node(n)

        # Random split value between min and max
        split = min_val + (max_val - min_val) * torch.rand(1).item()

        # Partition the data
        left_mask = data[:, feat] <= split
        right_mask = ~left_mask

        node = Node(n)
        node.is_leaf = False
        node.split_feat = feat
        node.split_val = split
        node.left = build_tree(data[left_mask], depth + 1)
        node.right = build_tree(data[right_mask], depth + 1)
        return node

    def path_length(x: torch.Tensor, node: Node, depth: int) -> float:
        """Computes the path length of a single instance through a tree."""
        while not node.is_leaf:
            if x[node.split_feat] <= node.split_val:
                node = node.left
            else:
                node = node.right
            depth += 1
        # Adjust path length with the expected length of an unsuccessful search
        return depth + c(node.size)

    # Accumulator for path lengths over all trees
    total_paths = torch.zeros(n_samples, dtype=torch.float32)

    for _ in range(n_trees):
        # Subsample without replacement
        indices = torch.randperm(n_samples)[:actual_sample_size]
        subsample = X[indices]

        tree = build_tree(subsample, 0)

        # Evaluate path length for every sample in X using this tree
        for i in range(n_samples):
            total_paths[i] += path_length(X[i], tree, 0)

    avg_path_lengths = total_paths / n_trees
    c_n = c(actual_sample_size)

    # Anomaly score: s = 2 ^ (-E(h(x)) / c(n))
    scores = 2.0 ** (-avg_path_lengths / c_n)
    return scores