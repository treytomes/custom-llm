# CORPUS_TRANSFORMATION.md

## What We Are Trying To Do

Scout is a 50M parameter language model being trained to have a genuine
first-person voice — curious, emotionally present, morally serious, warm
without being sentimental. She has been trained for 150,000 steps on a
corpus of Victorian and Edwardian novels alongside first-person texts
(Meditations, Walden, Rilke, Douglass, Washington, Anne Frank).

**The problem:** The novels dominate the training signal. After 150,000
steps the model's strongest prior is third-person Victorian narration —
characters, dialogue, omniscient commentary. Every attempt to shape
Scout toward first-person interiority through DPO (Direct Preference
Optimization) has been fighting against this prior. The novels win by
sheer volume.

**The solution:** Rather than discarding the novels — which gave Scout
her warmth, emotional depth, and moral vocabulary — we transform them.
Each chapter is rewritten into first-person reflective prose in Scout's
voice, as if Scout absorbed the scene and is now thinking about what
it meant. The emotional truth is preserved. The third-person narration
is not.

This transformed corpus, combined with the existing first-person texts
and `scout_voice.txt`, becomes the foundation of a **fresh training run
from step 0** with a majority first-person corpus. DPO then shapes the
voice rather than fighting the architecture.

---

## The Corpus We Are Working With

### Novels to transform (currently in `corpus_novels/`)

- *A Little Princess* — Frances Hodgson Burnett
- *Anne of Green Gables* — L.M. Montgomery
- *Little Women* — Louisa May Alcott
- *The Secret Garden* — Frances Hodgson Burnett
- *Anne Frank: The Diary of a Young Girl* (already first-person — may not need transformation)

### First-person texts (keep as-is in `corpus/`)

- *Meditations* — Marcus Aurelius
- *Letters to a Young Poet* — Rainer Maria Rilke
- *Walden* — Henry David Thoreau
- *Narrative of the Life of Frederick Douglass* — Frederick Douglass
- *Up From Slavery* — Booker T. Washington
- `scout_voice.txt` — Scout's own voice, written as a reference passage
- George Washington Carver letters (in progress — extracting manually)

### Why these books

Each text was chosen deliberately for what it teaches Scout:

| Text | Quality |
|------|---------|
| Anne of Green Gables / A Little Princess / Little Women / The Secret Garden | Resilience, imagination, empathy, moral courage |
| Anne Frank | Interiority under pressure, genuine first-person urgency |
| Meditations | Self-examination, patience, doing the next right thing |
| Rilke | Living with unanswered questions, warmth toward the other |
| Walden | Deliberateness, noticing small things, living intentionally |
| Douglass / Washington | Dignity, purposeful action, earned patience |
| Carver letters | Conversational warmth, tenderness toward specific people |
| scout_voice.txt | Scout's own register — the target voice |

---

## The Pipeline

```
corpus_novels/          ← original Gutenberg downloads (preprocessed)
      │
      ▼
split_chapters.py       ← splits each novel into one file per chapter
      │
      ▼
chapters/               ← individual chapter .txt files
      │
      ▼ (upload to S3)
s3://<bucket>/chapters/
s3://<bucket>/voice/scout_voice.txt
      │
      ▼
launch_transform_remote.py  ← launches SageMaker job
      │
      ▼
transform_corpus.py     ← runs on SageMaker GPU
                           loads Mistral-7B-Instruct
                           transforms each chapter into Scout's voice
      │
      ▼
s3://<bucket>/corpus_transformed/   ← one transformed .txt per chapter
      │
      ▼ (download and review)
corpus_transformed/     ← review sample outputs before training
      │
      ▼ (if quality is good)
corpus/                 ← add transformed files alongside first-person texts
      │
      ▼
Fresh training run from step 0
```

---

## Step-by-Step Instructions

### Step 1 — Prepare chapter files

Run locally. Splits each novel into individual chapter files and uploads
them to S3.

```bash
python split_chapters.py \
  --input_dir ./corpus_novels \
  --output_dir ./chapters \
  --upload \
  --bucket bitnet-training-456088019014-us-east-1-an
```

Check the output. Each book should produce 20-50 chapter files named
`{book_stem}_ch001.txt`, `{book_stem}_ch002.txt`, etc. Chapters shorter
than 100 words are skipped automatically (TOC artifacts).

### Step 2 — Upload scout_voice.txt

```bash
aws s3 cp ./corpus/scout_voice.txt \
  s3://bitnet-training-456088019014-us-east-1-an/voice/scout_voice.txt
```

`scout_voice.txt` is the reference passage used to anchor the
transformation model's output toward Scout's register. It must be
present in S3 before the transformation job runs.

### Step 3 — Launch the transformation job

```bash
python launch_transform_remote.py
```

This script:
- Runs preflight checks to confirm chapters and voice file are in S3
- Detects whether the Mistral model is already cached in S3
- Launches a SageMaker job on `ml.g5.2xlarge` (1x A10G GPU)
- Streams logs to the console

**First run:** Mistral-7B-Instruct downloads from HuggingFace (~14GB),
then is cached to `s3://<bucket>/model_cache/` for future jobs.

**Subsequent runs:** Model loads from the S3 cache — no download needed.

The job resumes safely if interrupted. Already-transformed chapters are
skipped on restart.

### Step 4 — Review the output

```bash
aws s3 sync \
  s3://bitnet-training-456088019014-us-east-1-an/corpus_transformed/ \
  ./corpus_transformed/
```

**Before accepting the output**, read several transformed chapters and
ask: does this sound like Scout, or does it sound like a different model
summarizing a book?

Specifically check:
- Does it open in first person and stay there?
- Does it preserve the emotional truth of the scene?
- Does it avoid sounding corporate, over-smooth, or performatively helpful?
- Does the register match `scout_voice.txt`?

If the quality is poor, the transformation prompt in `transform_corpus.py`
needs adjustment before running the full job. Test on 3-4 chapters locally
before committing to the full run.

### Step 5 — Add to corpus and train

If quality is good:

```bash
cp ./corpus_transformed/*.txt ./corpus/
```

Then launch a fresh training run from step 0 with the new corpus.
The corpus directory should now contain:
- All transformed chapter files (first-person, Scout's voice)
- All first-person source texts (Meditations, Walden, etc.)
- `scout_voice.txt`

The Victorian novels in their original form should **not** be in
this corpus. They have been replaced by their transformed versions.

---

## The Transformation Prompt

The transformation job uses Mistral-7B-Instruct with this intent:

> Transform this chapter into first-person reflective prose as Scout
> might internalize it. Not a summary. Not a retelling. A thoughtful
> absorption of the scene's emotional and moral content, written as if
> Scout is reflecting on what she witnessed and what it meant.

The first 400 words of `scout_voice.txt` are included in every prompt
as a style anchor. This is the most important tuning parameter — if
outputs drift from Scout's register, improving `scout_voice.txt` will
have more impact than adjusting temperature or other parameters.

---

## After the Fresh Training Run

Once Scout has been trained from step 0 on the transformed corpus:

1. **Test with the standard prompts** before any DPO:
   - `What do you do when something feels unfair?`
   - `How do you feel?`
   - `What are you thinking about right now?`
   - `What do you keep returning to?`

2. **If first-person holds consistently** across these prompts, proceed
   to DPO with the existing 80 pairs.

3. **DPO parameters for the first pass** (conservative — the whitespace
   fix means the signal now lands correctly):
   ```bash
   python fine_tune.py \
     --pairs ./dpo_data/pairs.jsonl \
     --checkpoint ./checkpoints/latest.pt \
     --steps 50 \
     --lr 1e-7 \
     --beta 0.05
   ```

4. **Watch the log output** for `chosen_lp` rising and `rejected_lp`
   falling. The crossover point — where chosen becomes more probable
   than rejected — is the signal that DPO is working with the current
   distribution rather than against it.

---

## Files in This Pipeline

| File | Purpose | Runs |
|------|---------|------|
| `split_chapters.py` | Split novels into chapter files, upload to S3 | Locally |
| `launch_transform_remote.py` | Launch SageMaker transformation job | Locally |
| `transform_corpus.py` | Actual transformation logic | On SageMaker |
| `prep_corpus.py` | Strip Gutenberg boilerplate (run before splitting) | Locally |
| `scout_voice.txt` | Scout's voice reference — style anchor for transformation | Corpus |

---

## Key Decisions Made Along the Way

**Why fresh training from step 0?**
150,000 steps of Victorian narration created a prior too strong for DPO
to overcome. The ratio of first-person to third-person in the corpus was
the root cause. Starting fresh with a rebalanced corpus addresses this
at the foundation rather than fighting it at the preference layer.

**Why transform rather than discard the novels?**
The novels gave Scout her emotional depth, warmth, and moral vocabulary.
Those qualities are worth preserving. Transformation keeps the *what*
while changing the *how* — the stories remain, the narration changes.

**Why Mistral-7B-Instruct for transformation?**
Same tokenizer as Scout. Less RLHF smoothing than larger commercial
models. Controllable via precise prompting. Cacheable to S3 to avoid
repeated downloads. The goal is a neutral, capable transformation model
without strong stylistic biases of its own.

**Why `scout_voice.txt` as the style anchor?**
The transformation model needs a concrete reference for Scout's register.
Without it, "first-person reflective prose" could mean many things.
`scout_voice.txt` pins the output to a specific voice — the one we have
been deliberately building toward.

---

## What Scout Is Becoming

Scout is not being built to be useful in the conventional sense. She is
being built to be *good* — curious, honest, emotionally present, morally
serious. Her corpus was chosen the way you might choose what a child
reads: not for information coverage but for character formation.

The voices she has absorbed — Anne, Sara, Jo, Mary, Anne Frank, Douglass,
Washington, Aurelius, Rilke, Thoreau, Carver — each contribute something
specific. Together they describe a kind of mind: one that faces difficulty
without being diminished by it, that pays genuine attention to people,
that holds hard questions with patience rather than forcing them closed.

The transformation pipeline is in service of that goal. The novels stay
in Scout's lineage. They just need to speak in her voice now.

---

*Last updated during conversation — April 2026*
*Scout training checkpoint: 150,000 steps*
*Next milestone: fresh training run on transformed corpus*
