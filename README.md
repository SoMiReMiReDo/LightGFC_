# LightGFC_
The code of paper " LIGHTWEIGHT GRAPH-FREE CONDENSATION WITH MLPDRIVEN OPTIMIZATION "

## Introduction
Graph condensation aims to compress large-scale graph data into a small-scale one, enabling efficient training of graph neural networks (GNNs) while preserving strong test performance and minimizing storage demands. Despite the promising performance of existing graph condensation methods, they still face two-fold challenges, i.e., bi-level optimization inefficiency & rigid condensed node label design, significantly limiting both efficiency and adaptability. To address such challenges, in this work, we propose a novel approach: LIGHTweight Graph-Free Condensation with MLP-driven optimization, named LIGHTGFC, which condenses large-scale graph data into a structure-free node set in a simple, accurate, yet highly efficient manner. Specifically, our proposed LIGHTGFC contains three essential stages: (S1) Proto-structural aggregation, which first embeds the structural information of the original graph into a proto-graph-free data through multihop neighbor aggregation; (S2) MLP-driven structural-free pretraining, which takes the proto-graph-free data as input to train an MLP model, aligning the structural condensed representations with node labels of the original graph; (S3) Lightweight class-to-node condensation, which condenses semantic and class information into representative nodes via a class-to-node projection algorithm with a lightweight projector, resulting in the final graph-free data. Extensive experiments show that the proposed LIGHTGFC achieves stateof-the-art accuracy across multiple benchmarks while requiring minimal training time (as little as 2.0s), highlighting both its effectiveness and efficiency.

## Requirements
All experiments are implemented in Python 3.9 with Pytorch 1.12.1.

```setup
conda env create -f environment.yml
```

## Datasets
* Cora and Citeseer, they can be downloaded from [PyG](https://www.pyg.org/).
* Ogbn-products can be downloaded from [OGB](https://ogb.stanford.edu/docs/nodeprop/).
* For Ogbn-arxiv, Flickr and Reddit, provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 

## Condensation and Model Training

To condense the graph using LightGFC and train GCN models:
```bash
$ python main.py --gpu 0 --dataset reddit --ratio 0.001 
```
And the result of exps will be saved in `./res/`.
