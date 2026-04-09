# Scout LLM

A lightweight research project for building and training a small (~50M parameter) language model from scratch.

The goal of this repository is to provide a simple, understandable pipeline for:

- preparing a text corpus
- training a transformer language model
- saving and resuming checkpoints
- testing the model with prompts
- running training locally or on AWS SageMaker

This repository is designed for experimentation and learning rather than production-scale training.

## Workflow

Usage: `python src/main.py <command>`

1. `corpus-generate-dialog` - Generate conversational training data using a teacher model.
2. `corpus-prepare` - Tokenize the corpus and report on the potential training stats. 
3. `train` - Initial training on the text corpus.
4. `dpo` - Generate DPO training pairs.
5. `build-dpo` - Convert the data collected in the interactive DPO process into a training set for fine-tuning.
6. `fine-tune` - Short, light training session on the DPO pairs to nudge Scout in a particular direction.
7. `chat` - Interactive loop for talking to Scout.
8. `dream` - Allow Scout to dream about the what happened in the last context window.

### TODO

* Fine-tune on the dream sequence.
* Allow Scout to write a diary entry on what happened during the last context window.
  * Use the diary entry for fine tuning and vector database recall.

