import numpy as np
import math
from collections import defaultdict

def beam_search_block_sharing(log_probs, beam_width, block_size, eos_token=-1):
    """
    Perform beam search decoding with memory-efficient block sharing.
    
    Args:
        log_probs: numpy array of shape (max_steps, vocab_size)
        beam_width: number of beams to maintain
        block_size: number of tokens per memory block
        eos_token: end-of-sequence token id (-1 for no early stopping)
    
    Returns:
        dict with keys: 'sequences', 'scores', 'total_blocks_allocated',
                        'blocks_in_use_final', 'naive_blocks_needed'
    """
    max_steps, vocab_size = log_probs.shape
    
    # ----- Block memory system -----
    class Block:
        __slots__ = ('tokens', 'ref_count')
        def __init__(self, tokens=None):
            self.tokens = tokens if tokens is not None else []
            self.ref_count = 1

    total_blocks_allocated = [0]   # mutable counter

    def new_block(tokens=None):
        total_blocks_allocated[0] += 1
        return Block(tokens)

    def share(block_list):
        """Return a new list referencing the same blocks, with ref_counts incremented."""
        new_list = block_list.copy()
        for blk in new_list:
            blk.ref_count += 1
        return new_list

    def release_blocks(block_list):
        """Decrement ref_counts of all blocks in the list."""
        for blk in block_list:
            blk.ref_count -= 1

    def append_token_inplace(block_list, token):
        """
        Append token to the sequence represented by block_list.
        Assumes the caller owns block_list and may modify it.
        Copy-on-write is triggered if the last block is shared (ref_count > 1).
        Returns the (possibly modified) block_list.
        """
        if not block_list:
            blk = new_block([token])
            block_list.append(blk)
            return block_list

        last = block_list[-1]
        if len(last.tokens) < block_size:
            if last.ref_count == 1:
                last.tokens.append(token)
            else:
                # copy-on-write
                new_tokens = last.tokens.copy()
                new_tokens.append(token)
                new_blk = new_block(new_tokens)
                last.ref_count -= 1
                block_list[-1] = new_blk
        else:
            # block full, allocate new one
            new_blk = new_block([token])
            block_list.append(new_blk)
        return block_list

    # ----- Step 0: initialise beams -----
    first_log = log_probs[0]
    top_indices = np.argsort(-first_log)[:beam_width]
    beams = []
    for idx in top_indices:
        score = first_log[idx]
        token = int(idx)
        blk = new_block([token])
        block_table = [blk]
        finished = (eos_token != -1 and token == eos_token)
        beams.append({
            'score': score,
            'block_table': block_table,
            'last_token': token,
            'finished': finished
        })
    beams.sort(key=lambda x: x['score'], reverse=True)

    # ----- Remaining steps -----
    for step in range(1, max_steps):
        active_beams = [b for b in beams if not b['finished']]
        finished_beams = [b for b in beams if b['finished']]
        if not active_beams:
            continue

        # 1. Generate all candidate expansions WITHOUT allocating blocks
        expansions = []   # (parent_idx, token, score, finished_flag)
        step_logs = log_probs[step]
        for p_idx, parent in enumerate(active_beams):
            base_score = parent['score']
            for token in range(vocab_size):
                score = base_score + step_logs[token]
                finished_flag = (eos_token != -1 and token == eos_token)
                expansions.append((p_idx, token, score, finished_flag))

        # 2. Combine with previous finished beams and sort
        items = []
        for fb in finished_beams:
            items.append((fb['score'], 'finished', fb))
        for exp in expansions:
            items.append((exp[2], 'expansion', exp))
        items.sort(key=lambda x: x[0], reverse=True)
        top_items = items[:beam_width]

        # 3. Build ordered placeholders and group selected expansions by parent
        ordered_beams = [None] * len(top_items)
        # map parent_idx -> list of (position, token, score, finished_flag)
        children_per_parent = defaultdict(list)

        for pos, (score, typ, data) in enumerate(top_items):
            if typ == 'finished':
                ordered_beams[pos] = data
            else:
                p_idx, token, sc, fin = data
                children_per_parent[p_idx].append((pos, token, sc, fin))

        # 4. For each parent with selected children, allocate blocks
        for p_idx, children in children_per_parent.items():
            parent = active_beams[p_idx]
            # "last child processed" = largest token (we iterated 0..vocab_size-1)
            max_tok = max(token for _, token, _, _ in children)

            # non-ownership children first (they must share)
            for pos, token, sc, fin in children:
                if token != max_tok:
                    shared = share(parent['block_table'])
                    new_blocks = append_token_inplace(shared, token)
                    ordered_beams[pos] = {
                        'score': sc,
                        'block_table': new_blocks,
                        'last_token': token,
                        'finished': fin
                    }

            # ownership child (can reuse parent's block table directly)
            for pos, token, sc, fin in children:
                if token == max_tok:
                    new_blocks = append_token_inplace(parent['block_table'], token)
                    ordered_beams[pos] = {
                        'score': sc,
                        'block_table': new_blocks,
                        'last_token': token,
                        'finished': fin
                    }
            # Parent's blocks are now fully transferred to its children
            parent['block_table'] = None

        # 5. Release blocks of any active parent that had NO children selected
        for p_idx, parent in enumerate(active_beams):
            if p_idx not in children_per_parent and parent['block_table'] is not None:
                release_blocks(parent['block_table'])
                parent['block_table'] = None

        # 6. Release blocks of previous beams that were dropped
        old_ids = set(id(b) for b in beams)
        keep_ids = set()
        for b in ordered_beams:
            if b is not None:
                keep_ids.add(id(b))
        for b in beams:
            if id(b) not in keep_ids and b['block_table'] is not None:
                release_blocks(b['block_table'])

        beams = ordered_beams

    # ----- Finalise output -----
    beams.sort(key=lambda x: x['score'], reverse=True)
    sequences = []
    scores = []
    for beam in beams[:beam_width]:
        tokens = []
        for blk in beam['block_table']:
            tokens.extend(blk.tokens)
        sequences.append(tokens)
        scores.append(round(beam['score'], 4))

    # Count distinct blocks still referenced by the final beams
    final_blocks = set()
    for beam in beams[:beam_width]:
        for blk in beam['block_table']:
            final_blocks.add(id(blk))

    naive_blocks_needed = beam_width * math.ceil(max_steps / block_size)
    scores_list = []
    for score in scores:
        scores_list.append(score.item())

    return {
        'sequences': sequences,
        'scores': scores_list,
        'total_blocks_allocated': total_blocks_allocated[0],
        'blocks_in_use_final': len(final_blocks),
        'naive_blocks_needed': naive_blocks_needed
    }