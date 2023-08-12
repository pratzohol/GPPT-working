# GPPT-working

- This repo contains code accompanying the paper, 	[GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks (Mingchen Sun et al., KDD 2022)](https://dl.acm.org/doi/abs/10.1145/3534678.3539249).

- I have covered the explanation of this paper here --> [How GPPT works?](https://pratzohol.github.io/MindML/Notes/Graph_Neural_Networks/icl-over-graphs-GPPT/). 

## Old Dependencies
[Original Source code](https://github.com/MingChen-Sun/GPPT) required the following:
* python 2.\* or python 3.\*
* PyTorch v1.8.+
* DGL v0.7+

## Requirement
The updated code uses the current following dependencies:
- python 3.9
- PyTorch v2.0.0+cu118
- DGL v1.1.1+cu118

## Code Explanation

### Overview of Modules

- `GPPT.py` : This is used to fine-tune the model on downstream node classfication task.
- `get_args.py` : This takes the argument from CLI for GPPT.py
- `model.py` : This contains backbone architecture of GraphSage for pre-training.
- `train_sampling_unsupervised.py` : This is used to pre-train the model using masked edge detection.
- `load_graph.py` and `utils.py` : This is used to load the graph dataset.
- `negative_sampler.py` : This is used to sample negative edges for masked edge detection.

### Pre-Training

- To pre-train the model, run the following command: `python train_sampling_unsupervised.py`
- CrossEntropyLoss is used to calculate loss for self-supervised task.

### Prompt and Fine-tuning

- For node classification task, run the following command: `python GPPT.py`
- First, load the parameters of pre-trained model.
- In the paper, they said for structure token 1-hop neighbour information will be aggregated using attention function. But, here they simply used mean function to aggregate the information and then concatenated the neighbour information embedding with node embedding of target node.
- Initialization of task token embeddings is same for all clusters. They used center node embedding of each class to initialize the task token embedding.
- The link prediction is done by finding cosine similarity between node embeddings. Thus, parameters $\phi$ of pre-trained projection head are non-existent. Only the parameters $\theta$ of backbone architecture and task token embeddings $E_1, \cdots, E_M$ are updated.
