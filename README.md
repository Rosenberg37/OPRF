# OPRF

This repository contains the code and resources for our paper:

Xueru Wen, Xiaoyang Chen, Xuanang Chen, Ben He, Le Sun. Offline Pseudo Relevance Feedback for Efficient and Effective Single-pass Dense Retrieval. In SIGIR 2023.

## Installation

Our code is developed largely depend on [Pyserini](https://github.com/castorini/pyserini/).
There are two ways you may set up the environment needed for run:

- install pip requirements

```shell
conda create --name PPRF python=3.8.15
conda activate PPRF
pip install -r requirements.txt
```

- install exported conda environment

```shell
conda env create -f conda.yaml
```

