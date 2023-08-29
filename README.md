# [RHGN: Relation-gated Heterogeneous Graph Network for Entity Alignment in Knowledge Graphs](https://aclanthology.org/2023.findings-acl.553/)

> Entity Alignment, which aims to identify equivalent entities from various Knowledge Graphs
(KGs), is a fundamental and crucial task in
knowledge graph fusion. Existing methods typically use triples or neighbor information to
represent entities, and then align those entities using similarity matching. Most of them,
however, fail to account for the heterogeneity
among KGs and the distinction between KG
entities and relations. To better solve these
problems, we propose a Relation-gated Heterogeneous Graph Network (RHGN) for entity
alignment in knowledge graphs. Specifically,
RHGN contains a relation-gated convolutional
layer to distinguish relations and entities in the
KG. In addition, RHGN adopts a cross-graph
embedding exchange module and a soft relation
alignment module to address the neighbor heterogeneity and relation heterogeneity between
different KGs, respectively. Extensive experiments on four benchmark datasets demonstrate
that RHGN is superior to existing state-of-theart entity alignment methods.

## Dataset
We use the entity alignment dataset OpenEA_15K_V1 in our experiments. OpenEA_15K_V1 can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA).

## Code
* "model.py" and "mynet_torch.py" are the implementations of RHGN.
* "layer_torch" is the implementation of RGC.

### Dependencies
* Python 3
* torch=1.12.1
* pytorch_geometric
* Scipy
* Numpy
* Pandas
* Scikit-learn

### Running

To run RHGN, simply run:
```
run.sh
```
> where *--input* is the datasets

## Citation
If you use our model or code, please kindly cite it as follows:      
```
@inproceedings{liu2023rhgn,
  title={RHGN: Relation-gated Heterogeneous Graph Network for Entity Alignment in Knowledge Graphs},
  author={Liu, Xukai and Zhang, Kai and Liu, Ye and Chen, Enhong and Huang, Zhenya and Yue, Linan and Yan, Jiaxian},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={8683--8696},
  year={2023}
}
```
