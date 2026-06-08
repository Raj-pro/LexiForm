# Paraphrase LLM — Optimization Guide

How to improve output quality **without adding new training pairs**, while preserving the user's hard constraint:

> Vary only **surface form** (synonyms, active↔passive voice). Keep **content words** intact: nouns, verbs, named entities, numbers, and emotion/sentiment.

---

## 1. Architecture Audit

Verified from [model/config.py](model/config.py), [model/blocks.py](model/blocks.py), [model/attention.py](model/attention.py), [model/model.py](model/model.py), [training/train.py](training/train.py), [training/loss.py](training/loss.py), [inference/infer.py](inference/infer.py).

| Aspect | Value |
|---|---|
| Total parameters | **13,539,329** (matches spec ~13.5M) |
| Tokenizer | SentencePiece BPE, vocab 16k, byte-fallback ON, prefix `<paraphrase>` |
| Architecture | T5-style encoder–decoder, 4+4 layers, d_model 256, 4 heads (head_dim 64), d_ff 1024, max_seq_len 128 |
| Positional encoding | RoPE on Q/K (no learned params) |
| Normalization | RMSNorm pre-norm, eps 1e-6 |
| FFN | SwiGLU, no bias |
| Weight tying | Output projection tied to input embedding (saves ~4.1M params) |
| Copy mechanism | Pointer-generator gate, log-domain blend via `logaddexp`, zero-init (p_gen ≈ 0.5 at start) |
| Dropout | 0.1 on embed, residuals, attention softmax |
| Train regularization | Label smoothing ε = 0.1, EMA decay 0.999, encoder-side 10% `<unk>` noise, bidirectional pair augmentation |
| Optimizer | AdamW (lr 3e-4, betas (0.9, 0.98), wd 0.01 on matmul only), linear warmup 200 → cosine to 1% lr, grad accum 4 (eff batch 256), grad clip 1.0, bf16 on CUDA |
| Inference | Beam search 12 beams + KV cache + no-repeat-3-gram + length penalty α=1.0 + semantic rerank (MiniLM-L6-v2 cosine − overlap penalty) |
| Phrase table | Bag-of-words n-gram → target-token log-probs ([build_phrase_table.py](build_phrase_table.py)), applied at decode via `_apply_phrase_bias()` at cross-attention peak |
| Final val loss | **2.83** at epoch 10 (plateaued after epoch 3–4 → dataset saturation, not model-capacity bound) |

### 1.1 Observed Failure Modes

Sampled from [output.txt](output.txt). Each illustrates a *violation of the preservation contract*:

| # | Failure mode | Example |
|---|---|---|
| F1 | **Number swap** | `$2.11, 11 percent` → `$1.63 or 8 percent` |
| F2 | **Named-entity injection** | source had no `PG&E`; output adds it |
| F3 | **Jurisdictional drift** | `state Supreme Court` → `U.S. Supreme Court` |
| F4 | **Context injection** | output prepends `"With the scandal hanging over Stewart's company..."` |
| F5 | **Negation flip** | `didn't go braless` → `mostly never wear a bra` |
| F6 | **Over-copy** | many outputs are near-identical to source (no rephrasing) |

**Verdict**: architecture is clean and modern — **no critical bugs**. The quality ceiling comes from (a) noisy training pairs and (b) decoding-time mechanism gaps that don't enforce the user's preservation rules. Both can be addressed without growing the dataset.

---

## 2. Optimization Strategy — Ranked by ROI

Five tiers. Tier 1 is purely inference-time (no retraining, no data changes) and should be implemented first.

### TIER 1 — Inference-time wins (no retraining)

#### 1.1 NER-locked copy mechanism *(fixes F1, F2, F3)*

Use spaCy NER + numeric/date regex to identify "must-not-change" source tokens. At each decode step, add a large positive bias to the **copy** distribution at those source positions, so the pointer overwhelmingly selects them.

Add to [inference/infer.py](inference/infer.py) near `_apply_phrase_bias` (~line 30):

```python
import re
import spacy
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp

NUMERIC_RE = re.compile(r"\b[\d][\d,\.]*\b|\$\d+|\d+%")

def build_lock_mask(src_text: str, tok) -> set[int]:
    """Return set of source TOKEN INDICES (in the encoded sequence) that must be copied."""
    doc = _get_nlp()(src_text)
    locked_chars = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
                          "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL", "NORP"}:
            locked_chars.update(range(ent.start_char, ent.end_char))
    for m in NUMERIC_RE.finditer(src_text):
        locked_chars.update(range(m.start(), m.end()))
    # Map char spans → token indices via SentencePiece offsets
    pieces = tok.sp.encode(f"<paraphrase> {src_text}", out_type=str)
    offsets, cur = [], len("<paraphrase> ")
    for p in pieces:
        clean = p.lstrip("▁")
        offsets.append((cur, cur + len(clean)))
        cur += len(clean) + (1 if p.startswith("▁") else 0)
    return {i for i, (a, b) in enumerate(offsets)
            if any(c in locked_chars for c in range(a, b))}

@torch.no_grad()
def apply_ner_copy_lock(
    log_probs: torch.Tensor,   # (num_beams, V), already log-domain
    attn_avg: torch.Tensor,    # (num_beams, 1, T_src)
    src_ids_b: torch.Tensor,   # (num_beams, T_src)
    locked_src_idx: set[int],
    boost: float = 8.0,        # log-domain boost ≈ +8 → e^8 ≈ 3000× preference
) -> None:
    if not locked_src_idx:
        return
    peak = attn_avg[:, 0, :].argmax(dim=-1)  # (num_beams,)
    for b in range(peak.size(0)):
        p = int(peak[b])
        if p in locked_src_idx:
            tok_id = int(src_ids_b[b, p])
            log_probs[b, tok_id] += boost  # in-place
```

Wire it into the beam-search step alongside the existing `_apply_phrase_bias` call.

Expected impact: eliminates the F1/F2/F3 class almost entirely. Cost: one spaCy doc parse per input (~5 ms CPU).

#### 1.2 POS-preserving reranker *(fixes F4, prevents content drift)*

Today the reranker scores only `cosine(src, cand) − 0.3·max(0, overlap − 0.85)`. Add a content-word fidelity term: the candidate must preserve the **ROOT verb** and the **set of head nouns** of the source. Modify `_rerank` in [inference/infer.py](inference/infer.py) (~line 369):

```python
def _content_anchor(text: str) -> tuple[str, frozenset[str]]:
    doc = _get_nlp()(text)
    root_verb = ""
    nouns = set()
    for sent in doc.sents:
        for tok in sent:
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                root_verb = tok.lemma_.lower()
            if tok.pos_ in {"NOUN", "PROPN"}:
                nouns.add(tok.lemma_.lower())
    return root_verb, frozenset(nouns)

def _content_penalty(src_text: str, cand_text: str) -> float:
    s_v, s_n = _content_anchor(src_text)
    c_v, c_n = _content_anchor(cand_text)
    verb_pen = 0.0 if s_v == c_v or not s_v else 0.4
    noun_miss = len(s_n - c_n) / max(1, len(s_n))
    return verb_pen + 0.5 * noun_miss

# inside _rerank, replace the score line with:
score = cos_sim - 0.3 * max(0.0, overlap - 0.85) \
                - 0.1 * abs(1 - len_ratio) \
                - _content_penalty(src_text, cand_text)
```

Expected impact: kills candidates that change the main verb or drop head nouns. Cost: one spaCy parse per candidate (~3 ms each).

#### 1.3 Sentiment-preservation gate *(serves the "emotion stays intact" rule)*

Reject candidates whose predicted sentiment differs from the source. Add near `get_reranker()`:

```python
from transformers import pipeline
_sent = None

def _get_sentiment():
    global _sent
    if _sent is None:
        _sent = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english",
                         top_k=1)
    return _sent

def sentiment_ok(src_text: str, cand_text: str, margin: float = 0.15) -> bool:
    p = _get_sentiment()([src_text, cand_text])
    s_lab, s_score = p[0][0]["label"], p[0][0]["score"]
    c_lab, c_score = p[1][0]["label"], p[1][0]["score"]
    if s_lab != c_lab and min(s_score, c_score) > margin:
        return False
    return True
```

Use it as a hard filter inside `_rerank` *before* sorting:

```python
candidates = [c for c in candidates if sentiment_ok(src_text, c)]
```

Cost: one mini-batch forward per inference call (~20 ms CPU for 8 candidates).

#### 1.4 Negation-polarity check *(fixes F5)*

Cheap, rule-based, no model:

```python
NEG_TOKENS = {"not", "n't", "never", "no", "without", "nor",
              "cannot", "can't", "won't", "didn't", "doesn't",
              "isn't", "aren't", "wasn't", "weren't", "fail", "lacks"}

def _neg_count(text: str) -> int:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for t in toks if t in NEG_TOKENS)

def negation_ok(src: str, cand: str) -> bool:
    return (_neg_count(src) % 2) == (_neg_count(cand) % 2)
```

Apply as a hard filter same as sentiment. Fixes the `didn't go braless` class instantly.

#### 1.5 Enable diverse beam search *(addresses F6, increases synonym variety)*

The plumbing already exists at [inference/infer.py:156-187](inference/infer.py#L156-L187) but defaults to `num_groups=1`, which disables it. Change CLI defaults:

```python
parser.add_argument("--num_groups",       type=int,   default=4)
parser.add_argument("--diversity_lambda", type=float, default=0.6)
```

`num_beams=12` split into 4 groups of 3 with a Hamming penalty between groups → forces lexical diversity per call. No code change beyond defaults.

#### 1.6 Upgrade reranker to a cross-encoder

`all-MiniLM-L6-v2` cosine similarity often misses subtle hallucinations because it pools sentence embeddings. A cross-encoder scores the (src, cand) pair jointly. Replace [inference/infer.py:62-67](inference/infer.py#L62-L67):

```python
from sentence_transformers import CrossEncoder

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading cross-encoder reranker...")
        _reranker = CrossEncoder("cross-encoder/stsb-roberta-large")
    return _reranker

# scoring (replaces cosine block):
pairs = [(src_text, c) for c in candidates]
sim_scores = get_reranker().predict(pairs)  # already 0..1
```

Cost: ~80 ms CPU per call for 8 candidates; ~10 ms on GPU. Worth it.

---

### TIER 2 — Phrase-table augmentation (no retraining, external lexicons)

#### 2.1 WordNet synonym injection *(directly serves the synonym-substitution goal)*

The current phrase table at [build_phrase_table.py](build_phrase_table.py) only knows substitutions seen in 62K pairs. Inject WordNet synsets to extend coverage. Add a `--wordnet` flag:

```python
from nltk.corpus import wordnet as wn

SPACY_TO_WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB,
                   "ADJ": wn.ADJ, "ADV": wn.ADV}

def wordnet_augment(table: dict, tok, default_logp: float = -4.0) -> None:
    nlp = _get_nlp()
    seen_unigrams = [k for k in table if len(k) == 1]
    for (tid,) in seen_unigrams:
        word = tok.decode([tid]).strip()
        if not word.isalpha():
            continue
        doc = nlp(word)
        if not doc or doc[0].pos_ not in SPACY_TO_WN_POS:
            continue
        wn_pos = SPACY_TO_WN_POS[doc[0].pos_]
        syns = {l.name().replace("_", " ")
                for s in wn.synsets(word, pos=wn_pos)
                for l in s.lemmas()
                if l.name().lower() != word.lower()}
        for syn in syns:
            syn_ids = tok.encode(syn)
            for sid in syn_ids:
                table[(tid,)].setdefault(sid, default_logp)
```

This biases the decoder toward **same-POS synonyms** at attention peaks — verbs stay verbs, nouns stay nouns, exactly matching the user's preservation rule.

Run: `python3 build_phrase_table.py --data data/clean.jsonl --tok tokenizer/tokenizer.model --out phrase_table.pkl --wordnet`

#### 2.2 Active↔passive voice transformer *(serves the active↔passive goal)*

New file `inference/voice_transform.py` — a deterministic post-processor that takes any decoded candidate and emits a voice-flipped twin, which is then thrown back into the rerank pool:

```python
import spacy
_nlp = spacy.load("en_core_web_sm")

def to_passive(sent: str) -> str | None:
    """Best-effort active→passive on a single clause. Returns None if not transformable."""
    doc = _nlp(sent)
    for s in doc.sents:
        subj = next((t for t in s if t.dep_ == "nsubj"), None)
        verb = next((t for t in s if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
        dobj = next((t for t in s if t.dep_ == "dobj"), None)
        if not (subj and verb and dobj):
            continue
        be = "was" if verb.tag_ in {"VBD", "VBN"} else "is"
        return f"{dobj.text.capitalize()} {be} {verb._.inflect('VBN') if hasattr(verb._,'inflect') else verb.lemma_+'ed'} by {subj.text}."
    return None

def to_active(sent: str) -> str | None:
    """Reverse of the above. Detect 'X was Yed by Z' and flip."""
    doc = _nlp(sent)
    for s in doc.sents:
        agent = next((t for t in s if t.dep_ == "agent"), None)
        if not agent:
            continue
        verb = agent.head
        pobj = next((c for c in agent.children if c.dep_ == "pobj"), None)
        subj = next((c for c in verb.children if c.dep_ == "nsubjpass"), None)
        if subj and pobj:
            return f"{pobj.text.capitalize()} {verb.lemma_}ed {subj.text}."
    return None
```

Wire into `_rerank`:

```python
extra = []
for c in candidates:
    for fn in (to_passive, to_active):
        v = fn(c)
        if v and v not in candidates:
            extra.append(v)
candidates += extra
```

Reranker then naturally picks the best. The grammar is approximate but works on the simple sentences this model targets (5–30 words).

---

### TIER 3 — Decoder hyperparameter tuning (no code, just sweeps)

Run [eval/evaluate.py](eval/evaluate.py) on the val set across:

| Param | Current | Sweep |
|---|---|---|
| `num_beams` | 12 | {12, 16, 20} |
| `length_penalty` | 1.0 | {0.6, 0.8, 1.0, 1.2} |
| `phrase_bias_beta` | 0.0 (off) | {0.2, 0.3, 0.5} |
| `no_repeat_ngram_size` | 3 | {3, 4} |
| `num_groups` × `diversity_lambda` | 1 × 0 | {(4, 0.4), (4, 0.6), (6, 0.6)} |

Optimize for the composite metric:
`J = BLEU(tgt) − 0.5·Self-BLEU(src) + 0.3·entity_preservation_rate`

(Self-BLEU penalty discourages near-copies; entity-preservation rewards the user's preservation rule.)

#### 3.1 Checkpoint averaging (Polyak)

A free quality bump. Drop this script next to [training/train.py](training/train.py):

```python
# tools/avg_ckpts.py
import torch, sys
paths = sys.argv[1:-1]; out = sys.argv[-1]
states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
avg = {k: sum(s["model_state"][k].float() for s in states) / len(states)
       for k in states[0]["model_state"]}
states[0]["model_state"] = avg
torch.save(states[0], out)
```

Run: `python3 tools/avg_ckpts.py checkpoints/checkpoint_epoch{8,9,10}.pt checkpoints/avg.pt`

Then evaluate `avg.pt` vs `best.pt`. Often +0.2–0.5 BLEU at zero cost.

---

### TIER 4 — Data cleanup (no new pairs added; one retrain)

The plateau at val loss 2.83 plus the F1–F5 errors strongly suggest noisy pairs in [data/clean.jsonl](data/clean.jsonl). Filtering bad pairs is **not** adding data — it should be done once.

Extend [data/clean.py](data/clean.py) with four new filters running on each `(src, tgt)` pair before write:

```python
import re, spacy
from transformers import pipeline

_nlp = spacy.load("en_core_web_sm")
_sent = pipeline("sentiment-analysis",
                 model="distilbert-base-uncased-finetuned-sst-2-english",
                 top_k=1)

NUM_RE = re.compile(r"\b\d[\d,\.]*\b|\$\d+|\d+%")
NEG = {"not", "n't", "never", "no", "without",
       "cannot", "can't", "won't", "didn't", "doesn't",
       "isn't", "aren't", "wasn't", "weren't"}

def entity_match(a: str, b: str) -> bool:
    ea = {(e.text.lower(), e.label_) for e in _nlp(a).ents}
    eb = {(e.text.lower(), e.label_) for e in _nlp(b).ents}
    return ea == eb

def number_match(a: str, b: str) -> bool:
    na = set(NUM_RE.findall(a))
    nb = set(NUM_RE.findall(b))
    return na == nb

def sentiment_match(a: str, b: str) -> bool:
    pa, pb = _sent([a, b])
    return pa[0]["label"] == pb[0]["label"]

def negation_match(a: str, b: str) -> bool:
    def cnt(s):
        return sum(1 for w in re.findall(r"[A-Za-z']+", s.lower()) if w in NEG)
    return (cnt(a) % 2) == (cnt(b) % 2)

def passes_preservation(src, tgt):
    return (entity_match(src, tgt) and number_match(src, tgt)
            and sentiment_match(src, tgt) and negation_match(src, tgt))
```

Then in the existing cleaning loop, skip pairs where `not passes_preservation(src, tgt)`.

**Expected reduction**: ~10–20% of pairs filtered (F1–F5 errors). Retrain once with the same config; expect val loss to drop into the 2.6–2.7 range and a sharp drop in deployed hallucination rate.

Retrain command (unchanged):

```bash
python3 -m training.train --data data/clean.jsonl --tok tokenizer/tokenizer.model
```

---

### TIER 5 — Lightweight model tweaks (no new data, no retrain)

#### 5.1 Copy-gate temperature

A scalar inference knob: multiply the copy-gate logit. `T_gate > 1` makes the model copy more, perfect for entity-rich inputs where Tier 1.1 is already locking entities — this hardens the bias.

In [model/model.py](model/model.py), modify the gate call in `_project_and_copy`:

```python
gate_logit = self.copy_gate(context, dec_state) * self.copy_gate_temp  # add attribute
```

Add `self.copy_gate_temp = 1.0` in `__init__`. Expose CLI flag in [inference/infer.py](inference/infer.py):

```python
parser.add_argument("--copy_gate_temp", type=float, default=1.0)
# after model loading:
model.copy_gate_temp = args.copy_gate_temp
```

Sweep `{0.8, 1.0, 1.5, 2.0}` on val. Higher T → more faithful, less paraphrased; tune jointly with Tier 1.5 diversity.

#### 5.2 Cross-attention temperature

Sharpening cross-attention makes the decoder track the source more tightly. In [model/attention.py](model/attention.py), in `CrossAttention.forward` where the manual softmax path lives (~line 158):

```python
attn = torch.softmax(scores / self.cross_attn_temp, dim=-1)  # add attribute, default 1.0
```

Add a `--cross_attn_temp` CLI knob. Values `< 1` sharpen; values `> 1` soften. Try `{0.7, 0.85, 1.0}`.

---

### TIER 0 — Guaranteed paraphrasing (output ≠ source, by construction)

Goal: make it **structurally impossible** for the model to emit a copy of the source. Tiers 1–5 increase preservation; Tier 0 is the dual — enforcing that some rephrasing always happens.

> **Caveat — unparaphrasable inputs:** short imperatives, vocatives, idioms, and emphatic phrases (`"oh raj"`, `"don't come"`, `"no means no"`, `"stop!"`) should be **echoed**, not paraphrased. Forcing rephrasing on them flips polarity or destroys the rhetorical force. Tier 0.0 below adds a *bypass gate* that suppresses Tiers 0.1–0.5 for these cases. Apply 0.0 first, then layer the rest.

#### 0.0 Unparaphrasable-input bypass (echo gate, no retrain)

A pre-encoder router. If any signal fires, return the input unchanged and skip Tiers 0.1–0.5 for this call. Place at the top of `paraphrase()` in [inference/infer.py:405-432](inference/infer.py#L405-L432):

```python
IDIOMS = {
    "no means no", "boys will be boys", "it is what it is",
    "que sera sera", "less is more", "rules are rules",
    "enough is enough", "fair is fair",
}

INTERJECTIONS = {"oh", "wow", "hey", "ouch", "ugh", "huh", "hmm",
                 "ah", "aha", "ow", "yay", "alas", "eek"}

def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

def is_unparaphrasable(text: str, locked_src_idx: set[int] | None = None,
                      src_tok_count: int | None = None) -> bool:
    t = text.strip()
    words = _word_tokens(t)

    # 1. Too short to rephrase while preserving content
    if len(words) < 4:
        return True

    # 2. Curated idiom set (case-insensitive, punctuation-stripped match)
    if _normalize(t) in {_normalize(x) for x in IDIOMS}:
        return True

    # 3. Pure interjection / vocative ("oh raj", "hey raj!")
    if words and words[0] in INTERJECTIONS and len(words) <= 3:
        return True

    # 4. Short imperative with negation — polarity carries the meaning
    NEG = {"not", "n't", "no", "never", "don't", "do", "stop", "cannot", "can't"}
    if len(words) <= 5 and any(w in NEG for w in words):
        return True

    # 5. Every non-stopword token is locked by Tier 1.1 NER/number lock
    #    → nothing unlocked is available to substitute
    if locked_src_idx is not None and src_tok_count is not None:
        # Stopword tokens are negligible; if locked covers ≥ 80% of non-special
        # source tokens, treat as fully locked.
        non_special = max(1, src_tok_count - 1)  # exclude <paraphrase> prefix
        if len(locked_src_idx) / non_special >= 0.8:
            return True

    return False
```

Wire into `paraphrase()`:

```python
def paraphrase(text, ...):
    locked = build_lock_mask(text, tok)                  # from Tier 1.1
    src_ids = torch.tensor([tok.encode("<paraphrase> " + text, ...)], ...)
    if is_unparaphrasable(text, locked, src_ids.size(1)):
        return [text]                                    # echo, skip beam search

    candidates = beam_search(model, src_ids, ..., locked_src_idx=locked)
    ranked = rerank(text, candidates, num_return=num_outputs)

    # Tier 0.1 identity reject — but ONLY for paraphrasable inputs
    src_norm = _normalize(text)
    ranked = [c for c in ranked if _normalize(c) != src_norm]
    if not ranked:
        ranked = [text]  # last-ditch echo rather than nonsense
    return ranked
```

#### 0.0.1 Confidence-based fallback (catches gate misses)

A safety net for inputs the gate failed to classify. After beam search, if **every** candidate falls below a quality floor (low cross-encoder similarity, high content drift from Tier 1.2, or polarity flip from Tier 1.4), echo the source. This means the model is allowed to say "I have no good paraphrase for this":

```python
def _is_safe_candidate(src: str, cand: str) -> bool:
    return (sentiment_ok(src, cand)            # Tier 1.3
        and negation_ok(src, cand)             # Tier 1.4
        and _content_penalty(src, cand) < 0.3) # Tier 1.2

# inside paraphrase(), after rerank:
safe = [c for c in ranked if _is_safe_candidate(text, c)]
if not safe:
    return [text]   # fallback: echo
ranked = safe
```

Combined effect: **0.0 + 0.0.1 + 0.1** guarantee that the model either emits a genuine paraphrase OR echoes the source — never produces a meaning-broken near-copy.

#### Updated stacking recommendation

| Want | Add |
|---|---|
| Echo trivial inputs, paraphrase the rest | 0.0 + 0.0.1 |
| + hard non-identity for paraphrasable inputs | 0.0 + 0.0.1 + 0.1 |
| + lexical-divergence floor | 0.0 + 0.0.1 + 0.1 + 0.2 + 0.5 |
| + model prefers rephrasing (retrain) | all of the above + 0.3 + 0.4 |

The order matters: 0.0 must run *before* 0.1's identity-reject, otherwise short imperatives get destroyed.



The current architecture has no such enforcement: the copy gate at [model/model.py:130-174](model/model.py#L130-L174) can place 100% of mass on copy at every step, and the no-repeat-3gram blocker at [inference/infer.py:286-294](inference/infer.py#L286-L294) only blocks repetition *within* the output, not against the source. The four modifications below close that gap, from cheapest to deepest.

#### 0.1 Source-identity hard reject (string-level guarantee, no retrain)

After beam search returns candidates, drop any that match the source under a normalized comparison. Falls back to the next-best beam, so no extra forward passes. Add to `paraphrase()` in [inference/infer.py:405-432](inference/infer.py#L405-L432):

```python
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

# inside paraphrase(), after `ranked = rerank(...)`:
src_norm = _normalize(text)
ranked = [c for c in ranked if _normalize(c) != src_norm]
if not ranked:
    # Last-ditch: any candidate that differs from source, even if reranker rejected it
    ranked = [c for c in candidates if _normalize(c) != src_norm][:num_outputs]
```

Hard guarantee: `output != source` modulo whitespace/punctuation/case. ~5 lines.

#### 0.2 Source n-gram block at decode (lexical-divergence guarantee, no retrain)

Stronger than 0.1: forbid the decoder from ever continuing a source-3-gram. This forces it to rephrase at every 3-token window, not just diverge somewhere. The existing `NgramState` class at [inference/infer.py:115-144](inference/infer.py#L115-L144) already implements per-beam incremental n-gram tracking — pre-seed it with the source n-grams:

```python
def _seed_source_ngrams(state: NgramState, src_ids: list[int], n: int) -> None:
    """Mark every (n-1)-prefix → next-token edge present in src_ids as banned."""
    if n <= 1:
        return
    for i in range(len(src_ids) - n + 1):
        prefix = tuple(src_ids[i : i + n - 1])
        nxt    = src_ids[i + n - 1]
        state.table.setdefault(prefix, set()).add(nxt)

# In beam_search(), when constructing the initial NgramStates (~line 256-261):
src_token_list = src_ids[0].tolist()
for t in topk_ids.tolist():
    s = NgramState(no_repeat_ngram)
    _seed_source_ngrams(s, src_token_list, no_repeat_ngram)  # NEW
    s.push(tok.bos_id)
    s.push(int(t))
    ngram_states.append(s)
```

Effect: any 3-token window from the source becomes a forbidden transition in the output. Combined with 0.1, you now have a guarantee that **no 3-gram from the source appears verbatim in the output**. Tune `n` ∈ {3, 4, 5} — smaller → stronger divergence, but harder to stay fluent.

Add a CLI knob `--source_block_n` so it can be ablated.

#### 0.3 Copy-budget cap (architectural — caps total copy mass per sequence)

Currently `p_gen` is free to be ≈0 at every step, meaning the decoder copies 100% of tokens. Add a *budget*: the cumulative copy probability summed across the output must not exceed a fraction `τ` of the sequence length. Implement as a soft constraint with a Lagrange multiplier during training, or as a hard clamp at inference.

**Inference-time clamp** (no retrain). Modify `_project_and_copy` in [model/model.py:130-174](model/model.py#L130-L174):

```python
def _project_and_copy(self, ..., copy_budget: torch.Tensor | None = None,
                                  copy_used:   torch.Tensor | None = None,
                                  max_copy_ratio: float = 0.5):
    ...
    # standard path:
    log_p_gen  = F.logsigmoid( p_gen_logit)
    log_p_copy = F.logsigmoid(-p_gen_logit)

    # NEW — hard cap. copy_used is (B, 1), running sum of p_copy from prior steps.
    if copy_budget is not None and copy_used is not None:
        # remaining budget per beam
        remaining = (copy_budget - copy_used).clamp_min(0.0)  # (B, 1)
        # Cap exp(log_p_copy) ≤ remaining. Subtract log-overshoot if any.
        p_copy = log_p_copy.exp()
        overshoot = (p_copy - remaining).clamp_min(0.0)
        log_p_copy = torch.log((p_copy - overshoot).clamp_min(1e-20))
        # Re-normalize log_p_gen so they sum to 1
        log_p_gen = torch.log((1 - (p_copy - overshoot)).clamp_min(1e-20))
    ...
```

Then in [inference/infer.py](inference/infer.py)'s beam loop, maintain `copy_used += p_copy` per beam per step, with `copy_budget = max_copy_ratio * src_len`. Default `max_copy_ratio = 0.5` means at most half of probability mass can come from the source copy distribution — forcing the vocab head to do work.

**Training-time soft variant** (one retrain). Add to `LabelSmoothingLoss` at [training/loss.py](training/loss.py):

```python
# Inside training loop, after model forward:
copy_mass = (-(F.logsigmoid(-p_gen_logit))).exp().mean()  # average p_copy
loss = loss + 0.1 * F.relu(copy_mass - 0.5)               # penalty if > 0.5
```

The model learns to keep copy mass under 50%, distributing the rest through the vocab distribution — which is where synonym substitution happens.

#### 0.4 Anti-copy contrastive loss (training-time, soft pressure away from identity)

Add a *negative* example during training: the source itself. The model should assign *low* probability to `y = x` and *high* probability to `y = paraphrase(x)`. Concretely, append to the loss in [training/train.py:156-160](training/train.py#L156-L160):

```python
with torch.autocast(...):
    scores  = model(src_ids, dec_input)
    loss    = criterion(scores, labels, input_is_log_probs=loss_in_log_probs)

    # Anti-copy term: feed source as if it were the target. We WANT this loss to be HIGH.
    src_as_tgt_input  = torch.cat([bos_col, src_ids[:, :-1]], dim=1)
    src_as_tgt_labels = src_ids
    neg_scores = model(src_ids, src_as_tgt_input)
    neg_loss   = criterion(neg_scores, src_as_tgt_labels,
                           input_is_log_probs=loss_in_log_probs)
    # Subtract — but only when neg_loss is "too low" (model assigns it too much prob).
    margin = 1.0
    loss = loss + 0.2 * F.relu(margin - (neg_loss - loss.detach()))
```

Interpretation: the loss-on-source must exceed the loss-on-target by at least `margin` nats per token. If the model is tempted to predict `y = x`, this term pushes back. Effectively a hinge contrastive loss between `(x, y)` (positive) and `(x, x)` (negative).

Cost: 2× forward passes per step → ~1.6× training time. Worth it for the structural guarantee.

#### 0.5 (Bonus) Minimum-edit-distance gate

After ranking, also require Levenshtein distance / source length ≥ τ_min (e.g. 0.2 → at least 20% of tokens differ). Pure post-filter:

```python
import difflib
def edit_ratio(a: str, b: str) -> float:
    ops = difflib.SequenceMatcher(None, a.split(), b.split()).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return edits / max(len(a.split()), 1)

ranked = [c for c in ranked if edit_ratio(text, c) >= 0.2]
```

Combines cleanly with 0.1 / 0.2.

#### Stacking recommendation

| Want | Add |
|---|---|
| Cheapest non-identity guarantee | 0.1 only |
| Lexical-divergence guarantee, no retrain | 0.1 + 0.2 + 0.5 |
| Train-once, model prefers rephrasing | 0.1 + 0.2 + 0.4 |
| Maximum guarantee, full retrain | 0.1 + 0.2 + 0.3 + 0.4 + 0.5 |

The first row alone gives a 100% guarantee `output != source` at zero training cost. The remaining tiers raise the *quality floor* of how different the output is.

#### Interaction with TIER 1.1 (NER-lock)

Tier 0.2 (source n-gram block) and Tier 1.1 (NER copy-lock) **conflict** on entity tokens — 1.1 wants to copy them, 0.2 wants to forbid the surrounding 3-grams. Resolve by exempting NER spans from the source-block table:

```python
# When seeding source-ngram block, skip n-grams that touch a locked span
for i in range(len(src_token_list) - n + 1):
    span = set(range(i, i + n))
    if span & locked_src_idx:   # from Tier 1.1
        continue
    ...
```

This keeps entities verbatim while still forcing all *non-entity* 3-grams to be rephrased — which is exactly the user's preservation contract.

---

### TIER 6 — Output fluency (post-decoder)

After running [sample.txt](sample.txt) through the current pipeline (Tier 0 enabled), the failure mode is clear: model emits tokenization artifacts (`thoughtsimus`, `.S.S`) and broken syntax. Tier 0 prevents copies but cannot manufacture fluency. Tier 6 adds a fluency stage *outside* the paraphraser.

> **Critical caveat:** a grammar-fix layer can repair surface form but **cannot restore meaning** that was lost during generation. If the source meaning is already gone from the beam output, "fixing" the grammar produces fluent-but-wrong text — arguably worse than visibly broken output, because it hides the failure. Prefer **rejecting** bad beams (6.1) over **polishing** them (6.2).

#### 6.1 Fluency-reject reranker (preferred)

Add a GPT-2 perplexity term to the reranker. Candidates with high perplexity are dropped, the Tier 0 fallback echoes the source if nothing survives. Honest about failure, never lies about meaning.

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math, torch

_lm = None
_lm_tok = None
def _get_lm():
    global _lm, _lm_tok
    if _lm is None:
        _lm_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        _lm     = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return _lm, _lm_tok

@torch.no_grad()
def perplexity(text: str) -> float:
    lm, tok = _get_lm()
    ids = tok(text, return_tensors="pt").input_ids
    if ids.size(1) < 2:
        return float("inf")
    out = lm(ids, labels=ids)
    return math.exp(out.loss.item())

# in rerank() scoring:
score = (cos_sim
         - 0.3 * max(0, overlap - 0.85)
         - 0.1 * abs(1 - len_ratio)
         - 0.05 * math.log1p(perplexity(cand)))
```

Cost: ~30 ms per candidate on CPU, ~5 ms on MPS/CUDA. GPT-2 small is 124M params (~500MB).

#### 6.2 Grammar-fix post-processor (use with caution)

Two variants. Both run *after* Tier 0 has selected the final output.

**Rule-based (LanguageTool):**

```python
import language_tool_python
_lt = language_tool_python.LanguageTool("en-US")
def grammar_fix(text: str) -> str:
    return _lt.correct(text)
```

Fixes punctuation, agreement, capitalization. Will not repair `"thoughtsimus"` (it's a "valid" unknown word to LT).

**Seq2seq (T5 grammar synthesis):**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
_gm = None
def _get_grammar_model():
    global _gm
    if _gm is None:
        tok = T5Tokenizer.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
        mdl = T5ForConditionalGeneration.from_pretrained(
            "pszemraj/flan-t5-large-grammar-synthesis"
        ).eval()
        _gm = (tok, mdl)
    return _gm

@torch.no_grad()
def grammar_fix(text: str) -> str:
    tok, mdl = _get_grammar_model()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
    out = mdl.generate(ids, max_length=256, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)
```

Cost: ~700MB model download, ~200 ms per call on CPU. **Will hide semantic failures behind fluent prose** — only enable when 6.1 rejects too aggressively and you'd rather have a plausible-sounding output than an echo.

#### 6.3 Joint paraphrase + fluency beam scoring

Most invasive but addresses the failure at *generation time* rather than after the fact. Inside `beam_search()` in [inference/infer.py](inference/infer.py), after computing the per-step log-probs:

```python
# At each step, score the candidate prefix under GPT-2.
# Cache prefix → score to avoid quadratic re-computation.
for b in range(num_beams):
    text_so_far = tok.decode(beam_tokens[b][1:])  # skip BOS
    cum[b] += 0.2 * (-perplexity_logp_per_token(text_so_far))
```

Cost: turns beam search from ~50ms to ~500ms per input. Worth it only if 6.1 alone isn't enough.

---

### TIER 7 — Replace the base model

The blunt truth that emerged from running [sample.txt](sample.txt): a 13.5M-param model trained on 63K news pairs cannot paraphrase literary prose. No amount of decoder surgery changes that. If you want quality > 4/10 on out-of-domain inputs, **swap the engine**.

| Pretrained model | Params | Strengths | Drop-in path |
|---|---|---|---|
| `humarin/chatgpt_paraphraser_on_T5_base` | 220M | trained on ChatGPT paraphrases, diverse outputs | wrap in HF `T5ForConditionalGeneration`, keep Tier 0/1 wrappers |
| `prithivida/parrot_paraphraser_on_T5` | 220M | adequacy-fluency-diversity scored | similar |
| `tuner007/pegasus_paraphrase` | 568M | strongest at preserving semantics | similar |
| `Vamsi/T5_Paraphrase_Paws` | 220M | trained on PAWS, robust to negation | similar |

Integration sketch (replaces `model.generate()` / beam_search call only — Tier 0 reranking still applies):

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class HFParaphraser:
    def __init__(self, name="humarin/chatgpt_paraphraser_on_T5_base"):
        self.tok = T5Tokenizer.from_pretrained(name)
        self.mdl = T5ForConditionalGeneration.from_pretrained(name).eval()

    @torch.no_grad()
    def generate(self, text, num_beams=8, num_return=5):
        ids = self.tok(f"paraphrase: {text}", return_tensors="pt",
                       truncation=True, max_length=256).input_ids
        out = self.mdl.generate(ids, num_beams=num_beams,
                                num_return_sequences=num_return,
                                max_length=256, repetition_penalty=1.2)
        return [self.tok.decode(o, skip_special_tokens=True) for o in out]
```

Tier 0 / 1 / 2 / 6 all still apply on top of this — they're decode-side, model-agnostic. The pretrained model fills the gap that retraining the current 13.5M cannot.

This is **not** "adding training pairs" — it's swapping in a model someone else already trained.

---

## 3. Verification Plan

### 3.1 New metrics to add to [eval/evaluate.py](eval/evaluate.py)

```python
def entity_preservation(src: str, hyp: str) -> bool:
    ea = {(e.text.lower(), e.label_) for e in _nlp(src).ents}
    eh = {(e.text.lower(), e.label_) for e in _nlp(hyp).ents}
    return ea.issubset(eh)  # hyp may add nothing, must keep all src entities

def number_preservation(src: str, hyp: str) -> bool:
    return set(NUM_RE.findall(src)).issubset(set(NUM_RE.findall(hyp)))

def sentiment_preservation(src: str, hyp: str) -> bool:
    pa, pb = _sent([src, hyp])
    return pa[0]["label"] == pb[0]["label"]
```

Aggregate as percentages over the val set. Add to the report line.

### 3.2 Before/after protocol

For each tier you apply:

1. Run `python3 -m eval.evaluate --ckpt checkpoints/best.pt --tok tokenizer/tokenizer.model`.
2. Record: `BLEU`, `ROUGE-L`, `BERTScore-F`, `copy-rate`, `Self-BLEU`, plus the three new metrics.
3. Compute composite `J = BLEU − 0.5·Self-BLEU + 0.3·(entity_pres + number_pres + sentiment_pres)/3`.
4. Manually inspect 50 outputs and tally: synonym used? voice changed? all facts preserved?

### 3.3 Targets after Tier 1 + 2

| Metric | Baseline (approx) | Target |
|---|---|---|
| Entity preservation | ~70% | **≥ 95%** |
| Number preservation | ~75% | **≥ 98%** |
| Sentiment preservation | ~85% | **≥ 92%** |
| Self-BLEU vs source | unknown | **≤ 0.65** (proves rephrasing) |
| BLEU vs target | baseline | **no regression** |

---

## 4. Quick-Start Checklist (ordered)

1. `pip install spacy nltk transformers && python -m spacy download en_core_web_sm && python -m nltk.downloader wordnet` — install deps.
2. Add Tier 1.4 (negation check) and 1.5 (diverse beam defaults) — both are < 30 lines, biggest immediate diversity gain.
3. Add Tier 1.1 (NER-lock) — biggest hallucination reduction.
4. Add Tier 1.2 + 1.3 (POS reranker, sentiment gate).
5. Add Tier 1.6 (cross-encoder reranker).
6. Add Tier 2.1 (WordNet) — rebuild `phrase_table.pkl`; pass `--phrase_table phrase_table.pkl --phrase_bias_beta 0.3` at inference.
7. Add Tier 2.2 (voice transformer).
8. Run Tier 3 sweep; pick best decoder config.
9. Run Tier 3.1 checkpoint averaging; A/B vs `best.pt`.
10. If still unsatisfied, run Tier 4 (data cleanup) + one retrain. If you want zero retraining, also try Tier 5 knobs (`--copy_gate_temp`, `--cross_attn_temp`).

---

## 5. Out of Scope (explicit)

- Adding new paraphrase training pairs from any source.
- Growing the model (layers, d_model, heads, vocab).
- Changing the tokenizer.
- Architectural changes (RoPE, RMSNorm, SwiGLU stay).

---

## 6. Why This Matches Your Goal

| Your requirement | Mechanism that enforces it |
|---|---|
| Don't paraphrase the unparaphrasable | Tier 0.0 echo gate + Tier 0.0.1 confidence fallback |
| Output ≠ source (guaranteed, when paraphrasable) | Tier 0.1 identity reject + Tier 0.2 source n-gram block |
| Model *prefers* rephrasing | Tier 0.3 copy-budget + Tier 0.4 anti-copy loss |
| Synonyms allowed | Tier 2.1 WordNet injection + Tier 1.5 diverse beam |
| Active↔passive allowed | Tier 2.2 voice transformer |
| Nouns preserved | Tier 1.1 NER-lock + Tier 1.2 POS reranker + Tier 4 entity filter |
| Verbs preserved | Tier 1.2 ROOT-verb penalty + Tier 4 entity filter |
| Emotion preserved | Tier 1.3 sentiment gate + Tier 4 sentiment filter |
| Numbers/dates preserved | Tier 1.1 numeric regex lock + Tier 4 number filter |
| Negation preserved | Tier 1.4 polarity check + Tier 4 negation filter |
| No new training pairs | Every tier obeys this. Tier 4 only *removes* bad pairs. |
