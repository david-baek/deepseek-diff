# Towards Understanding Distilled Reasoning Models: A Representational Approach

In this paper, we investigate how model distillation impacts the development of reasoning features in large language models (LLMs). To explore this, we train a crosscoder on Qwen-series models and their fine-tuned variants. Our results suggest that the crosscoder learns features corresponding to various types of reasoning, including self-reflection and computation verification. Moreover, we observe that distilled models contain unique reasoning feature directions, which could be used to steer the model into over-thinking or incisive-thinking mode. In particular, we perform analysis on four specific reasoning categories: (a) self-reflection, (b) deductive reasoning, (c) alternative reasoning, and (d) contrastive reasoning. Finally, we examine the changes in feature geometry resulting from the distillation process and find indications that larger distilled models may develop more structured representations, which correlate with enhanced distillation performance. By providing insights into how distillation modifies the model, our study contributes to enhancing the transparency and reliability of AI systems.

Download the trained crosscoders [here](https://www.dropbox.com/scl/fo/n5ekjy5c6e7gmm2djkqxn/AFyp_3ncFup_L7MINkRMCKw?rlkey=0vhcr8r4te9k2zjvjhiiafqyh&e=1&st=l7r4cwdi&dl=0).

Paper: [arXiv](https://arxiv.org/abs/2503.03730)

We have adapted the [open-source implementation](https://github.com/ckkissane/crosscoder-model-diff-replication) of crosscoder for our experiments.
