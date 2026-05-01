def deduplicate_recomputation(nodes: list) -> dict:
    """
    Remove recomputation nodes that alias already-seen storage.

    Args:
        nodes: list of dicts with keys 'id', 'storage_ptr', 'cost'

    Returns:
        dict with 'kept_ids' (list[int]) and 'total_cost_saved' (float)
    """
    kept_ids = []
    kept_ptrs = []
    total_cost_saved = 0.0
    for node in nodes:
        nid,ptr,cost=node["id"],node["storage_ptr"],node["cost"]
        if kept_ptrs.count(ptr) > 0:
            total_cost_saved += cost
        else:
            kept_ids.append(nid)
            kept_ptrs.append(ptr)
    return {"kept_ids":kept_ids,"total_cost_saved":total_cost_saved}