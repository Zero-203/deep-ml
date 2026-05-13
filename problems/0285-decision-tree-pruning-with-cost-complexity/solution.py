def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    # Your code here
    alpha_list = []
    def cal_node(node: dict, alpha_list: list) -> (int,int):
        if node["left"] is None and node["right"] is None:
            return 1,node["errors"]
        else:
            left_cnt,left_error=cal_node(node["left"],alpha_list)
            right_cnt,right_error=cal_node(node["right"],alpha_list)
            alpha_list.append((node["errors"]-left_error-right_error)/(left_cnt+right_cnt-1))
            return left_cnt+right_cnt,left_error+right_error
    if tree is not None:
        cal_node(tree,alpha_list)
    alpha_list.sort()
    return alpha_list