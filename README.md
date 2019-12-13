# MulT

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)  

> Pytorch implementation for the paper "[Multimodal Transformer for Unaligned Multimodal Language Sequences](https://arxiv.org/pdf/1906.00295.pdf)". 

> Original author's implementation is [here](https://github.com/yaohungt/Multimodal-Transformer).
 
  
## Datasets

 - Data files (containing processed MOSI, MOSEI and IEMOCAP datasets) can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0).

 - To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).


## Prerequisites
- Python 3.6
- [Pytorch (>=1.0.0) and torchvision](https://pytorch.org/)
- CUDA 10.0 or above


## Run the Code

1. Create (empty) folders for data and pre-trained models:
~~~~
mkdir data pre_trained_models
~~~~

and put the downloaded data in 'data/'.

2. Command as follows
~~~~
python main.py [--FLAGS]
~~~~

Note that the defualt arguments are for unaligned version of MOSEI. For other datasets, please refer to Supplmentary.

### Results

```
nohup python main.py &
```

 - unaligned version of MOSEI

Output: [nohup.out](https://github.com/Eurus-Holmes/MulT/blob/master/nohup.out)

```
MAE:  0.6139981
Correlation Coefficient:  0.6773945850196033
mult_acc_7:  0.48873148744365746
mult_acc_5:  0.5028976175144881
F1 score:  0.8201431177436439
Accuracy:  0.8200330214639515
```

### If Using CTC

Transformer requires no CTC module. However, as we describe in the paper, CTC module offers an alternative to applying other kinds of sequence models (e.g., recurrent architectures) to unaligned multimodal streams.

If you want to use the CTC module, plesase install warp-ctc from [here](https://github.com/baidu-research/warp-ctc).

The quick version:
~~~~
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
export WARP_CTC_PATH=/home/xxx/warp-ctc/build
~~~~

## Acknowledgement
Some portion of the code were adapted from the [fairseq](https://github.com/pytorch/fairseq) repo.


## Citation

```tex
@inproceedings{tsai2019multimodal,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Liang, Paul Pu and Kolter, J Zico and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
