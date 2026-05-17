import numpy as np

def simulate_lock_free_sync(
    n_workers: int,
    flag_size: int,
    worker_ops: list,
    max_rounds: int = 1000
) -> dict:
    """
    Simulate lock-free synchronization among workers using shared flags.

    Args:
        n_workers:   Number of workers (0 to n_workers-1)
        flag_size:   Size of the shared flag array (initialized to zeros)
        worker_ops:  List of lists; worker_ops[i] contains the ordered
                     operations for worker i.  Each operation is a tuple:
                       ("set",  flag_index, value)
                       ("wait", flag_index, value)
                       ("work", worker_id, task_id)
        max_rounds:  Maximum simulation rounds before halting

    Returns:
        dict with keys:
          "completed" : list of (worker_id, task_id) tuples
          "flags"     : list of ints, final flag array
          "deadlock"  : bool
    """
    flags = [0] * flag_size
    workers_idx = [0] * n_workers
    completed = []
    remain = 0

    for widx in range(n_workers):
        while workers_idx[widx]<len(worker_ops[widx]):
            work = worker_ops[widx][workers_idx[widx]]
            op = work[0]
            if op == "work":
                remain += 1
            workers_idx[widx]+=1
    
    workers_idx = [0] * n_workers

    for epoch in range(max_rounds):
        done = 0
        for widx in range(n_workers):
            if workers_idx[widx]>=len(worker_ops[widx]):
                continue
            work = worker_ops[widx][workers_idx[widx]]
            op = work[0]
            if op == "set":
                flag_idx,val=work[1],work[2]
                flags[flag_idx]=val
                workers_idx[widx]+=1
                done += 1
            elif op == "wait":
                flag_idx,val=work[1],work[2]
                if flags[flag_idx] == val:
                    workers_idx[widx]+=1
                    done += 1
            elif op == "work":
                completed.append((work[1],work[2]))
                workers_idx[widx]+=1
                done += 1
                remain -= 1
            else:
                print("Error op",op)
        if all(workers_idx[w] >= len(worker_ops[w]) for w in range(n_workers)):
            return {"completed": completed, "flags": flags, "deadlock": False}
        if done == 0:
            return {"completed":completed,"flags":flags,"deadlock":True}
    return {"completed":completed,"flags":flags,"deadlock":False}
