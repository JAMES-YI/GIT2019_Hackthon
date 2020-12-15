# -*- coding: utf-8 -*-
"""
This file is an implementation of singular value thresholding algorithm 
for matrix completion which is introduced in [1].

[1] Cai JF, Candès EJ, Shen Z. A singular value thresholding algorithm for matrix completion. 
SIAM Journal on optimization. 2010 Mar 3;20(4):1956-82.

Created on Fri Oct 11 14:50:54 2019

@author: jiryi
"""

import numpy as np
import numpy.linalg as npla
import random 

if __name__=="__main__":
    
    RowNum = 500
    ColNum = 500
    r = 5
    Ml = np.random.normal(0,1,[RowNum,r])
    Mr = np.random.normal(0,1,[ColNum,r])
    M = np.matmul(Ml,np.transpose(Mr))
    MVec = M.reshape([-1,])
    # npla.matrix_rank(M)
    
    ObsvNum = int(0.6*RowNum*ColNum)
    ObsvSet = random.sample(range(RowNum*ColNum),ObsvNum)
    MObsvVec = np.zeros([RowNum*ColNum,])
    MObsvVec[ObsvSet] = MVec[ObsvSet]
    MObsv = MObsvVec.reshape([RowNum,ColNum])
    
    Delta = 1.2*RowNum*ColNum
    Eps = 0.0001
    MaxIte = 1000
    Tau = 5*RowNum
    Inc = 5
    k = Tau / (Delta*npla.norm(MObsv))
    
    Y = k*delta*MObsv
    
    
    
    
