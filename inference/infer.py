"""
Run paraphrase inference (KV-cached, batched beam search).
Run: python3 -m inference.infer --ckpt checkpoints/best.pt --tok tokenizer/tokenizer.model
"""
import argparse
import difflib
import pickle
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from model.config import ModelConfig
from model.model import ParaphraseModel, _build_src_pad_mask
from tokenizer.tokenizer import Tokenizer


_reranker = None


# ---------- Tier 0: guarantee output ≠ source (when paraphrasable) ----------
#
# 0.0 — echo gate: identify inputs that should NOT be rephrased and short-
#       circuit beam search. Short imperatives, vocatives, idioms, etc.
# 0.1 — identity reject: drop any candidate whose normalized form matches src.
# 0.2 — source-n-gram block: pre-seed NgramState with src n-grams so the
#       decoder cannot continue any source n-gram (see _seed_source_ngrams).
# 0.5 — edit-distance gate: drop candidates with too few token-level edits.
#
# All four are decode-time only — no retraining, no extra deps.

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE    = re.compile(r"\s+")
_WORD_RE  = re.compile(r"[A-Za-z']+")

IDIOMS = {
    "no means no", "boys will be boys", "it is what it is",
    "que sera sera", "less is more", "rules are rules",
    "enough is enough", "fair is fair",
}
INTERJECTIONS = {
    "oh", "wow", "hey", "ouch", "ugh", "huh", "hmm",
    "ah", "aha", "ow", "yay", "alas", "eek",
}
NEG_WORDS = {
    "not", "n't", "no", "never", "don't", "do", "stop",
    "cannot", "can't", "won't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't",
}


def _normalize(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", s.lower())).strip()


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def is_unparaphrasable(text: str) -> bool:
    """Heuristic gate — when True, skip paraphrasing and echo the input.

    Signals (any one fires → echo):
      • < 4 content words ("oh raj", "stop!")
      • Matches a curated idiom set ("no means no")
      • Starts with an interjection and ≤ 3 words ("hey raj")
      • ≤ 5 words and contains negation ("don't come")

    NER-saturation (Tier 1.1) is deliberately omitted here — that check needs
    spaCy and lives with the rest of the Tier-1 entity machinery.
    """
    t = text.strip()
    words = _word_tokens(t)
    if len(words) < 4:
        return True
    if _normalize(t) in {_normalize(x) for x in IDIOMS}:
        return True
    if words and words[0] in INTERJECTIONS and len(words) <= 3:
        return True
    if len(words) <= 5 and any(w in NEG_WORDS for w in words):
        return True
    return False


def edit_ratio(a: str, b: str) -> float:
    """Token-level edit ratio. 0.0 = identical word sequences, ~1.0 = disjoint."""
    a_toks = a.split()
    b_toks = b.split()
    ops = difflib.SequenceMatcher(None, a_toks, b_toks).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return edits / max(len(a_toks), 1)


# ---------- phrase table ----------

def load_phrase_table(path: str) -> dict | None:
    """Load a phrase table built by build_phrase_table.py, or return None."""
    if not path:
        return None
    with open(path, "rb") as f:
        table = pickle.load(f)
    print(f"Phrase table loaded: {len(table):,} n-gram entries from {path}")
    return table


@torch.no_grad()
def _apply_phrase_bias(
    scores: torch.Tensor,       # (num_beams, V) — modified in-place
    attn_avg: torch.Tensor,     # (num_beams, 1, T_src)
    src_ids_b: torch.Tensor,    # (num_beams, T_src)
    table: dict,
    beta: float,
) -> None:
    """Add β × log_p_table[best_matching_src_ngram] to scores, in-place.

    For each beam, take the source position with the highest cross-attention
    weight, find the longest matching n-gram ending at that position (n=3→1),
    look it up in the table, and scatter-add the sparse bias vector.  No hit
    → beam is untouched.
    """
    num_beams, _, T_src = attn_avg.shape
    device = scores.device
    peak_pos = attn_avg[:, 0, :].argmax(dim=-1)  # (num_beams,)

    for b in range(num_beams):
        p = int(peak_pos[b])
        for n in (3, 2, 1):
            start = max(0, p - n + 1)
            ngram = tuple(src_ids_b[b, start : p + 1].tolist())
            hit = table.get(ngram)
            if hit is not None:
                idx = torch.tensor(list(hit.keys()),   device=device, dtype=torch.long)
                val = torch.tensor(list(hit.values()), device=device, dtype=scores.dtype)
                scores[b].index_add_(0, idx, val * beta)
                break  # use longest match only


def get_reranker() -> SentenceTransformer:
    global _reranker
    if _reranker is None:
        print("Loading semantic reranker...")
        _reranker = SentenceTransformer("all-MiniLM-L6-v2")
    return _reranker


def load_model(ckpt_path: str, tok_path: str, device: torch.device):
    tok  = Tokenizer(tok_path)
    # Checkpoints contain only tensors + plain dicts/strings/ints — safe under
    # weights_only=True, which blocks the arbitrary-code-execution path
    # weights_only=False keeps open via pickle.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ModelConfig(**ckpt["config"])
    model  = ParaphraseModel(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, config


# ---------- cache plumbing ----------

def _expand_caches(caches: list[dict], num_beams: int) -> None:
    """Replicate the batch dimension of all cached K/V tensors from 1 -> num_beams."""
    for layer in caches:
        for kind in ("self", "cross"):
            c = layer[kind]
            if "k" in c:
                c["k"] = c["k"].expand(num_beams, *c["k"].shape[1:]).contiguous()
                c["v"] = c["v"].expand(num_beams, *c["v"].shape[1:]).contiguous()


def _reorder_caches(caches: list[dict], parent: torch.Tensor) -> None:
    for layer in caches:
        for kind in ("self", "cross"):
            c = layer[kind]
            if "k" in c:
                c["k"] = c["k"].index_select(0, parent)
                c["v"] = c["v"].index_select(0, parent)


# ---------- no-repeat n-gram ----------
#
# Per-beam incremental state. Each beam carries:
#   table:  dict[(n-1)-prefix -> set of seen next tokens]
#   suffix: the current (n-1)-token suffix (tuple), or None if prefix < n-1
# Banned-tokens lookup is O(1) per beam per step; the table is updated by one
# entry per step. When beams are reordered by `parent`, child i clones
# parent[i]'s table (shallow copy of the dict, then we'll mutate it on the
# next step's append).


class NgramState:
    __slots__ = ("n", "table", "suffix")

    def __init__(self, n: int):
        self.n = n
        self.table: dict[tuple, set[int]] = {}
        self.suffix: tuple = ()

    def clone(self) -> "NgramState":
        c = NgramState(self.n)
        # Deep-copy the sets so two child beams from the same parent can diverge.
        c.table  = {k: set(v) for k, v in self.table.items()}
        c.suffix = self.suffix
        return c

    def banned(self) -> set[int]:
        if self.n <= 0 or len(self.suffix) < self.n - 1:
            return set()
        return self.table.get(self.suffix, set())

    def push(self, token: int) -> None:
        # If we have a full (n-1)-prefix already, record (prefix -> token).
        if self.n > 1 and len(self.suffix) == self.n - 1:
            self.table.setdefault(self.suffix, set()).add(token)
            self.suffix = self.suffix[1:] + (token,)
        else:
            # Growing the suffix up to length n-1.
            self.suffix = self.suffix + (token,)
            if self.n > 1 and len(self.suffix) > self.n - 1:
                self.suffix = self.suffix[-(self.n - 1):]


def _seed_source_ngrams(state: NgramState, src_ids: list[int], n: int) -> None:
    """Tier 0.2 — pre-populate the n-gram block table with every (n-1)-prefix
    → next-token edge present in the source. After seeding, the decoder cannot
    continue any source n-gram, forcing rephrasing at every window of size n.
    """
    if n <= 1 or len(src_ids) < n:
        return
    for i in range(len(src_ids) - n + 1):
        prefix = tuple(src_ids[i : i + n - 1])
        nxt    = src_ids[i + n - 1]
        state.table.setdefault(prefix, set()).add(nxt)


# ---------- length penalty ----------

def _length_penalty(length: int, alpha: float) -> float:
    # Google NMT formula — better behaved than `length ** alpha` for short outputs.
    return ((5.0 + length) / 6.0) ** alpha


# ---------- beam search ----------

def _grouped_topk(
    scores: torch.Tensor,     # (B, V) cum log-probs per (beam, token)
    num_beams: int,
    num_groups: int,
    diversity_lambda: float,
    V: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diverse beam search top-k: split beams into G groups, apply a Hamming
    penalty to tokens already chosen by prior groups at this timestep.
    Returns (top_lp, top_idx) where top_idx encodes parent*V + token, same as
    plain top-k on the flattened (B, V) tensor.
    """
    bpg = num_beams // num_groups
    device = scores.device
    chosen: set[int] = set()
    lp_parts, idx_parts = [], []
    for g in range(num_groups):
        sl = slice(g * bpg, (g + 1) * bpg)
        cum_g = scores[sl].clone()
        if chosen:
            penalty = torch.zeros(V, device=device, dtype=cum_g.dtype)
            penalty[torch.tensor(sorted(chosen), device=device, dtype=torch.long)] = diversity_lambda
            cum_g = cum_g - penalty.unsqueeze(0)
        flat_g = cum_g.view(-1)
        lp_g, idx_in_g = flat_g.topk(bpg)
        parent_in_g = torch.div(idx_in_g, V, rounding_mode="floor")
        token_g     = idx_in_g % V
        abs_parent  = parent_in_g + g * bpg
        lp_parts.append(lp_g)
        idx_parts.append(abs_parent * V + token_g)
        chosen.update(token_g.tolist())
    return torch.cat(lp_parts), torch.cat(idx_parts)


@torch.no_grad()
def beam_search(
    model: ParaphraseModel,
    src_ids: torch.Tensor,             # (1, T_src)
    tok: Tokenizer,
    config: ModelConfig,
    num_beams: int = 12,
    max_new_tokens: int = 128,
    no_repeat_ngram: int = 3,
    length_penalty: float = 1.0,
    num_return: int = 8,
    num_groups: int = 1,
    diversity_lambda: float = 0.5,
    phrase_table: dict | None = None,
    phrase_beta: float = 0.2,
    block_source_ngrams: bool = True,
) -> list[str]:
    assert src_ids.size(0) == 1, "beam_search expects a single source at a time"
    if num_groups > 1:
        assert num_beams % num_groups == 0, (
            f"num_beams ({num_beams}) must be divisible by num_groups ({num_groups})"
        )
    device = src_ids.device

    encoder_out   = model.encode(src_ids)                           # (1, T_src, d)
    src_attn_mask = _build_src_pad_mask(src_ids, config.pad_id, dtype=encoder_out.dtype)
    caches        = model.init_caches()

    # Step 0: feed BOS, expand top-k into num_beams beams.
    cur = torch.full((1, 1), tok.bos_id, dtype=torch.long, device=device)
    scores0, attn0 = model.decode_step(cur, encoder_out, src_attn_mask, src_ids, caches, step=0)
    # scores0: (1, 1, V) — log-probs (copy) or logits; normalize to log-probs.
    scores0 = scores0[0, -1]                                        # (V,)
    if not config.use_copy:
        scores0 = F.log_softmax(scores0, dim=-1)

    # Phrase-table bias at step 0 (single beam, src_ids shape (1, T_src)).
    if phrase_table is not None and attn0 is not None:
        _apply_phrase_bias(scores0.unsqueeze(0), attn0[0].unsqueeze(0), src_ids, phrase_table, phrase_beta)
        scores0 = scores0.squeeze(0)

    if num_groups > 1:
        # Single root beam — apply DBS by picking each group's top tokens with
        # the Hamming penalty against earlier groups.
        bpg = num_beams // num_groups
        chosen: set[int] = set()
        lp_parts, id_parts = [], []
        for g in range(num_groups):
            s = scores0.clone()
            if chosen:
                penalty = torch.zeros_like(s)
                penalty[torch.tensor(sorted(chosen), device=device, dtype=torch.long)] = diversity_lambda
                s = s - penalty
            lp_g, id_g = s.topk(bpg)
            lp_parts.append(lp_g)
            id_parts.append(id_g)
            chosen.update(id_g.tolist())
        topk_lp  = torch.cat(lp_parts)
        topk_ids = torch.cat(id_parts)
    else:
        topk_lp, topk_ids = scores0.topk(num_beams)                 # (num_beams,)

    beam_scores  = topk_lp.clone()                                  # (num_beams,)
    beam_tokens: list[list[int]] = [[tok.bos_id, int(t)] for t in topk_ids.tolist()]
    beam_alive   = torch.ones(num_beams, dtype=torch.bool, device=device)

    # Incremental per-beam n-gram state, seeded with [BOS, first_token].
    # Tier 0.2: if block_source_ngrams, also pre-seed with every source n-gram
    # so the decoder can never continue a source n-gram in its output.
    src_token_list = src_ids[0].tolist() if block_source_ngrams else None
    ngram_states: list[NgramState] = []
    for t in topk_ids.tolist():
        s = NgramState(no_repeat_ngram)
        if src_token_list is not None:
            _seed_source_ngrams(s, src_token_list, no_repeat_ngram)
        s.push(tok.bos_id)
        s.push(int(t))
        ngram_states.append(s)

    # Replicate everything we need across beams.
    _expand_caches(caches, num_beams)
    encoder_out   = encoder_out.expand(num_beams, *encoder_out.shape[1:]).contiguous()
    src_attn_mask = src_attn_mask.expand(num_beams, *src_attn_mask.shape[1:]).contiguous()
    src_ids_b     = src_ids.expand(num_beams, -1).contiguous()

    cur = topk_ids.view(num_beams, 1)
    V   = config.vocab_size

    completed: list[tuple[float, list[int]]] = []

    for step in range(1, max_new_tokens):
        # Single batched decoder step over all live beams.
        scores, attn_avg = model.decode_step(cur, encoder_out, src_attn_mask, src_ids_b, caches, step=step)
        scores = scores[:, -1, :]                                   # (num_beams, V)
        if not config.use_copy:
            scores = F.log_softmax(scores, dim=-1)

        # Phrase-table bias: nudge toward trained substitutions at the
        # attended source position, before n-gram blocking and top-k.
        if phrase_table is not None and attn_avg is not None:
            _apply_phrase_bias(scores, attn_avg[:, -1:, :], src_ids_b, phrase_table, phrase_beta)

        # no-repeat n-gram blocking per beam (incremental O(1) state lookup)
        if no_repeat_ngram > 0:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for b in range(num_beams):
                banned = ngram_states[b].banned()
                if banned:
                    idx = torch.tensor(list(banned), device=device, dtype=torch.long)
                    mask[b].index_fill_(0, idx, True)
            scores = scores.masked_fill(mask, float("-inf"))

        # Dead beams have already been recorded in `completed`; their
        # `beam_scores` are forced to -inf below (after eos handling), so any
        # cum-score they produce here is -inf and they cannot win top-k.
        cum = beam_scores.unsqueeze(1) + scores                     # (num_beams, V)
        if num_groups > 1:
            top_lp, top_idx = _grouped_topk(cum, num_beams, num_groups, diversity_lambda, V)
        else:
            flat = cum.view(-1)
            top_lp, top_idx = flat.topk(num_beams)
        parent = torch.div(top_idx, V, rounding_mode="floor")
        token  = top_idx %  V

        parent_list = parent.tolist()
        token_list  = token.tolist()
        new_beam_tokens = [beam_tokens[p] + [t] for p, t in zip(parent_list, token_list)]
        # Clone parent's n-gram state into each child, then push the new token.
        new_ngram_states = []
        for p, t in zip(parent_list, token_list):
            s = ngram_states[p].clone()
            s.push(t)
            new_ngram_states.append(s)

        # Reorder caches by parent so they line up with the new beams.
        _reorder_caches(caches, parent)

        # Update bookkeeping.
        beam_scores  = top_lp
        beam_tokens  = new_beam_tokens
        ngram_states = new_ngram_states
        beam_alive   = beam_alive.index_select(0, parent)

        # Move newly completed beams (those that emitted EOS) into the completed set.
        eos_mask = (token == tok.eos_id) & beam_alive
        if eos_mask.any():
            for b in eos_mask.nonzero(as_tuple=False).flatten().tolist():
                ids = beam_tokens[b]
                ln  = max(1, len(ids) - 1)  # exclude BOS
                completed.append((float(beam_scores[b]) / _length_penalty(ln, length_penalty), ids))
            beam_alive = beam_alive & ~eos_mask

        # Evict dead beams from future top-k consideration. With beam_scores =
        # -inf, any (beam_scores + scores) row for that beam is -inf and the
        # slot is free to be re-occupied by a child of a live parent next step.
        beam_scores = beam_scores.masked_fill(~beam_alive, float("-inf"))

        if not beam_alive.any() or len(completed) >= num_beams:
            break

        cur = token.view(num_beams, 1)

    # Finalize any still-alive beams.
    for b in beam_alive.nonzero(as_tuple=False).flatten().tolist():
        ids = beam_tokens[b]
        ln  = max(1, len(ids) - 1)
        completed.append((float(beam_scores[b]) / _length_penalty(ln, length_penalty), ids))

    completed.sort(key=lambda x: x[0], reverse=True)

    results: list[str] = []
    seen: set[str] = set()
    for _, ids in completed:
        clean = [i for i in ids if i not in (tok.bos_id, tok.eos_id, tok.pad_id)]
        text = tok.decode(clean).strip()
        if text and text not in seen:
            seen.add(text)
            results.append(text)
            if len(results) >= num_return:
                break
    return results


# ---------- reranking ----------

def rerank(source: str, candidates: list[str], num_return: int) -> list[str]:
    """
    Score each candidate by:
      - semantic similarity to source (want HIGH — meaning preserved)
      - word overlap with source       (want LOW — actually rephrased)
      - length ratio to source         (want CLOSE to 1 — penalise truncation /
                                        over-expansion that drops or invents content)
    Combined score = similarity
                     - 0.3 * max(0, overlap   - 0.85)
                     - 0.1 * |1 - len_ratio|
    """
    if not candidates:
        return []

    reranker = get_reranker()
    src_emb  = reranker.encode([source],      convert_to_tensor=True)  # (1, d)
    cand_emb = reranker.encode(candidates,    convert_to_tensor=True)  # (N, d)

    sim = util.cos_sim(src_emb, cand_emb)[0]  # (N,)

    src_tokens = set(source.lower().split())
    src_len    = max(1, len(source.split()))
    scored: list[tuple[float, str]] = []
    for i, cand in enumerate(candidates):
        cand_tokens = cand.lower().split()
        if len(cand_tokens) < 4:
            continue
        overlap     = sum(1 for t in cand_tokens if t in src_tokens) / len(cand_tokens)
        length_pen  = abs(1.0 - len(cand_tokens) / src_len)
        score = float(sim[i]) - 0.3 * max(0.0, overlap - 0.85) - 0.1 * length_pen
        scored.append((score, cand))

    scored.sort(reverse=True)
    return [c for _, c in scored[:num_return]]


def paraphrase(
    text: str,
    model: ParaphraseModel,
    tok: Tokenizer,
    config: ModelConfig,
    device: torch.device,
    num_outputs: int = 3,
    num_beams: int = 12,
    num_groups: int = 1,
    diversity_lambda: float = 0.5,
    phrase_table: dict | None = None,
    phrase_beta: float = 0.2,
    block_source_ngrams: bool = True,
    min_edit_ratio: float = 0.15,
    echo_short_inputs: bool = True,
) -> list[str]:
    # Tier 0.0 — echo gate: don't paraphrase short imperatives / idioms /
    # vocatives. Bypasses the entire pipeline.
    if echo_short_inputs and is_unparaphrasable(text):
        return [text]

    src     = "<paraphrase> " + text
    src_ids = torch.tensor(
        [tok.encode(src, max_length=config.max_seq_len)],
        dtype=torch.long, device=device,
    )
    candidates = beam_search(
        model, src_ids, tok, config,
        num_beams=num_beams, num_return=10,
        num_groups=num_groups, diversity_lambda=diversity_lambda,
        phrase_table=phrase_table, phrase_beta=phrase_beta,
        block_source_ngrams=block_source_ngrams,
    )
    # Over-rank then post-filter so Tier 0.1 / 0.5 have multiple options to
    # choose from instead of seeing only the reranker's top-N.
    ranked = rerank(text, candidates, num_return=max(num_outputs * 4, 10))

    # Tier 0.1 — drop any candidate whose normalized form equals the source.
    src_norm = _normalize(text)
    ranked = [c for c in ranked if _normalize(c) != src_norm]

    # Tier 0.5 — drop candidates with insufficient token-level edits.
    if min_edit_ratio > 0:
        ranked = [c for c in ranked if edit_ratio(text, c) >= min_edit_ratio]

    # Fallback: if nothing survived the Tier-0 filters, echo the source rather
    # than emit a broken near-copy. This is the safety net for paraphrasable
    # inputs that the model couldn't actually paraphrase well.
    if not ranked:
        return [text]

    return ranked[:num_outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="checkpoints/best.pt")
    parser.add_argument("--tok",    default="tokenizer/tokenizer.model")
    parser.add_argument("--n",      default=3,  type=int)
    parser.add_argument("--beams",  default=12, type=int)
    parser.add_argument("--groups", default=1,  type=int,
                        help="Diverse beam search groups (must divide --beams). "
                             "Set >1 (e.g. 3) for paraphrase-style lexical diversity.")
    parser.add_argument("--diversity_lambda", default=0.5, type=float,
                        help="Hamming penalty subtracted from a token's score "
                             "for each prior group that chose it this timestep.")
    parser.add_argument("--phrase_table", default="", type=str,
                        help="Path to phrase_table.pkl built by build_phrase_table.py. "
                             "Omit to run without the phrase-table bias.")
    parser.add_argument("--phrase_beta", default=0.2, type=float,
                        help="Strength of the phrase-table additive log-prob bias (0 = off).")
    # Tier 0 — guaranteed-paraphrasing knobs.
    parser.add_argument("--no_block_source", action="store_true",
                        help="Disable Tier 0.2 source-n-gram blocking at decode.")
    parser.add_argument("--min_edit_ratio", default=0.15, type=float,
                        help="Tier 0.5 — drop candidates with token-edit ratio < this. "
                             "0 disables.")
    parser.add_argument("--no_echo_gate", action="store_true",
                        help="Disable Tier 0.0 echo gate (short imperatives / idioms / "
                             "vocatives are normally returned unchanged).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, config = load_model(args.ckpt, args.tok, device)
    ptable = load_phrase_table(args.phrase_table)
    get_reranker()

    print("Paraphrase model ready. Type a sentence (Ctrl+C to quit).\n")
    while True:
        try:
            text = input("Input: ").strip()
            if not text:
                continue
            outputs = paraphrase(text, model, tok, config, device,
                                 num_outputs=args.n, num_beams=args.beams,
                                 num_groups=args.groups,
                                 diversity_lambda=args.diversity_lambda,
                                 phrase_table=ptable, phrase_beta=args.phrase_beta,
                                 block_source_ngrams=not args.no_block_source,
                                 min_edit_ratio=args.min_edit_ratio,
                                 echo_short_inputs=not args.no_echo_gate)
            for i, o in enumerate(outputs, 1):
                print(f"  [{i}] {o}")
            print()
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
