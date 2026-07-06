## Secrecy

Can you interact with an LLM such that you and the LLM provider retain essential information? This repository contains experiments testing this question by restating as a model training characteristics problem.

### Quickstart

Most experiments in this repository require the use of a somewhat recent GPU, and has been tested with CUDA version 13.0 and Nvidia drivers >=580. For currently unsupported devices (V100s, P100s etc.) with recent Pytorch library versions you may have to modify the uv.lock file for use with a compiled version that matches your device.

To get started, clone and navigate to `secrecy`, which contains numerous experimental training pipelines, model architectures, dataset generators, and utilities.


#### Tag-based secrecy

For tag-based secrecy, start by accessing a trained causal language model and tokenizer, download and tokenize and batch the appropriate datasets if necessary. Then generate the inversion dataset (input tokens and their mid-way embeddings) via

`uv run python inversion_dataset_generation`

This will generate a pyarrow `datasets` object, which you can then use to train an inversion model via (replace `<n>` with the number of GPUs on your node)

`uv run torchrun --nproc_per_node=<n> secret_decoder.py`

With this trained, now train a secret (encoder) model dataset via

`uv run torchrun --nproc_per_node=<n> overfit_secret_tag_generation.py`

and train the secret decoder via

`uv run torchrun --nproc_per_node=<n> overfit_secret_decoder.py`


#### Built-in secrecy



### TODOs:

[ ] add utilities for preprocessed dataset generation
[ ] embedding tensorification optimization from zero-copy pyarrow