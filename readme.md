# Robust Ensemble Clustering Using Probability Trajectories

## Overview
This repository provides the MATLAB code for two ensemble clustering algorithms, namely, `probability trajectory accumulation (PTA)`
and `probability trajectory based graph partitioning (PTGP)`. If you find the code useful for your research,please cite the paper below.   

```
Dong Huang, Jian-Huang Lai, and Chang-Dong Wang. 
Robust Ensemble Clustering Using Probability Trajectories, 
IEEE Transactions on Knowledge and Data Engineering, 2016, 28(5), pp.1312-1326.
```

## Description of Files
There are mainly two types of materials in this directory:
1. A pool of 200 base clusterings for each dataset;  
2. The code of the PTA and PTGP algorithms.

### Data

The base clustering pools for the ten datasets are provided in the following MAT files:

```
bc_pool_MF.mat
bc_pool_IS.mat
bc_pool_MNIST.mat
bc_pool_ODR.mat
bc_pool_LS.mat
bc_pool_PD.mat
bc_pool_USPS.mat
bc_pool_FC.mat
bc_pool_KDD99_10P.mat
bc_pool_KDD99.mat
```

There are two variables in the MAT file for each dataset, namely, members and gt. The variable gt is the ground-truth label, which is an N-dimension vector. The variable members is an N x s matrix, where each column of it is a candidate base clustering.


### Code

The file entitled 'demo_PTA_and_PTGP.m' is the main file for running PTA and PTGP. You may change the following settings in order to test the performance of PTA and PTGP:

```
1) dataName:	the dataset to be used.
2) M:		the ensemble size.
3) cntTimes:	run PTA and PTGP for cntTimes times and obtain the average performance.
4) para.K:	the parameter K.
5) para.T:	the parameter T.
6) clsNums:	a vector of positve integers, specifying different numbers of clusters for PTA and PTGP.
```

The execution results and the variable 'bcIdx' will be saved in results.mat. The bcIdx is a cntTimes x M matrix and stores the information of the ensembles. Each row in bcIdx includes M indices for choosing base clusterings in the pool and thus represents an ensemble of M base clusterings. When comparing our approach to other approaches, please make sure that they use the same base clustering settings, i.e., use the ensembles generated by the same 'bcIdx'.

## Further Questions?
Don't hesitate to contact me if you have any questions regarding this work.  
Email: huangdonghere at gmail dot com  
Website: https://www.researchgate.net/publication/284259332  

	
