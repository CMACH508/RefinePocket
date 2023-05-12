# RefinePocket: An Attention-Enhanced and Mask-Guided Deep Learning Approach for Protein Binding Site Prediction



This repository contains the source code, trained models and the test sets for RefinePocket.



## Introduction

Protein binding site prediction is an important prerequisite task of drug discovery and design. While binding sites are very small, irregular and varied in shape, making the prediction very challenging. Standard 3D U-Net has been adopted to predict binding sites but got stuck with unsatisfactory prediction results, incomplete, out-of-bounds, or even failed. The reason is that this scheme is less capable of extracting the chemical interactions of the entire region and hardly takes into account the difficulty of segmenting complex shapes. In this paper, we propose a refined U-Net architecture, called RefinePocket, consisting of an attention-enhanced encoder and a mask-guided decoder. During encoding, taking binding site proposal as input, we employ Dual Attention Block (DAB) hierarchically to capture rich global information, exploring residue relationship and chemical correlations in spatial and channel dimensions respectively. Then, based on the enhanced representation extracted by the encoder, we devise Refine Block (RB) in the decoder to provide self-guidance and refine uncertain regions progressively, resulting in more precise segmentation. Experiments show that DAB and RB complement and promote each other, making RefinePocket has an average improvement of 10.02% on DCC and 4.26% on DVO compared with the state-of-the-art method on four test sets.



<img src=".\figs\overview.jpg" width="100%"/>



## Dataset

**Train data:** Train dataset ``scPDB`` can be downloaded from here (http://bioinfo-pharma.u-strasbg.fr/scPDB/).

**Test data:** Test datasets can be downloaded according to the following links, `COACH420` (https://github.com/rdk/p2rank-datasets/tree/master/coach420),  `HOLO4k` (https://github.com/rdk/p2rank-datasets/tree/master/holo4k), `SC6K` (https://github.com/devalab/DeepPocket), `PDBbind` (http://www.pdbbind.org.cn/download.php).



## Data processing

For COACH420, HOLO4K and SC6K, the preprocessing procedure is the same as in [DeepPocket](https://github.com/devalab/DeepPocket). For PDBbind, the refined set of version 2020 is used in our experiments, in which proteins with more than 50% sequence identity to those in ScPDB are removed to avoid data leakage.

The test datasets preprocessed by us can be downloaded from https://zenodo.org/record/7669660#.Y_dUyS-KEys.



## Train and test

### Train

To train RefinePocket from scratch, run the following command:

```
python train.py
```

The details of the parameters are specified in the code.



### Test

Download the pretrained_model.pth.tar to ``checkpoints/`` from https://zenodo.org/record/7669660#.Y_dUyS-KEys. 

To test RefinePocekt on COACH420 in terms of DCC and DVO, run the following command:

```
python test.py --test_set coach420
```

To test RefinePocekt on COACH420 in terms of DCA top-n, run the following command:

```
python test.py --test_set coach420 --is_dca 1 --rank 0
```

To test RefinePocekt on COACH420 in terms of DCA top-n+2, run the following command:

```
python test.py --test_set coach420 --is_dca 1 --rank 2
```

To test RefinePocekt on HOLO4K and SC6K, modify the parameter  --test_set.

------------------------------------------------------------------------------------------------------------------------------------------

To test RefinePocekt on PDBbind in terms of DCC and DVO, run the following command:

```
python test_pdbbind.py --test_set pdbbind
```

To test RefinePocekt on PDBbind in terms of DCA top-n, run the following command:

```
python test_pdbbind.py --test_set pdbbind --is_dca 1 --rank 0
```

To test RefinePocekt on PDBbind in terms of DCA top-n+2, run the following command:

```
python test_pdbbind.py --test_set pdbbind --is_dca 1 --rank 2
```

