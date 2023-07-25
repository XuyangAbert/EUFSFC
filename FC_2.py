# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:11:01 2019

@author: xyan
"""

import numpy as np
import math
import pandas as pd
import numpy.matlib as b
from sklearn.preprocessing import normalize
import time
from entropy_estimators import *
start = time.time()
def Input():
    # Read the data from the txt file
    sample = pd.read_csv('colon.csv',header=None)
    (N, L) = np.shape(sample)
    dim = L - 1
    label1 = sample.iloc[:,L-1]
    label = label1.values
    data = sample.iloc[:,0:dim]
    NewData = Pre_Data(data)
    return NewData,label
def Pre_Data(data):
    [N,L] = np.shape(data)
    NewData = np.zeros((N,L))
    for i in range(L):
        Temp = data.iloc[:,i]
        if np.max(Temp) == 0:
            NewData[:,i] = np.zeros((N,1))
        else:
            Temp = (Temp - np.min(Temp))/(max(Temp)-min(Temp))
            NewData[:,i] = Temp      
    return NewData
                
def Distribution_Est(data, dim):
    DC_mean = np.zeros(dim)
    DC_std = np.zeros(dim)
    for i in range(dim):
        TempClass = data[:,i]
        DC_mean[i] = np.mean(TempClass)
        DC_std[i] = np.std(TempClass)
    return DC_mean,DC_std

def Feature_Dist(data,dim):
    Dist = []
    DisC = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            DisC[i,j] = Sym_Cal(data,i,j)
            DisC[j,i] = DisC[i,j]
            Dist.append(DisC[i,j])
    return DisC,Dist

def KLD_Cal(data,i,j,Var,Corr):
    Var1 = Var[i]
    Var2 = Var[j]
    P = Corr[i,j]
    Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
    D_KL = Sim / (Var1 + Var2)
    return D_KL 

def Sym_Cal(data,i,j):
    I_ij = midd(data[:,i],data[:,j])
    H_I = entropyd(data[:,i])
    H_J = entropyd(data[:,j])
    D_KL = 1 - 2*(I_ij)/(H_I + H_J)
    return D_KL

def fitness_cal(DisC, DC_means, DC_std, data, StdF, gamma):
    fitness = np.zeros(len(DC_means))
    # print(np.shape(fitness))
    for i in range(len(DC_means)):
        TempSum = 0
        for j in range(len(DC_means)):
            if j != i:
                D = DisC[i,j]
                TempSum = TempSum + (math.exp(- (D**2) / StdF))**gamma
        fitness[i] = TempSum
    return fitness

def Pseduo_Peaks(DisC, Dist, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    
    # The temporal sample space in terms of mean and standard deviation
    sample = np.vstack((DC_Mean,DC_Std)).T
    # Spread= np.max(Dist)
    # Search Stage of Pseduo Clusters at the temporal sample space
    NeiRad = 0.7*np.max(Dist)
    # NeiRad = (StdF/gamma)
    i = 0
    marked = []
    C_Indices = np.arange(1, len(DC_Mean)+1) # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:
        PeakIndices.append(np.argmax(F))
        Pfitness.append(np.max(F))
        indices = NeighborSearch(DisC, data, sample, PeakIndices[i], marked, NeiRad)
        C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices=[PeakIndices[i]]
        co.append(len(indices)) # Number of samples belong to the current 
        # identified pseduo cluster
        marked = np.concatenate(([marked,indices]))
        # Fitness Proportionate Sharing
        F = Sharing(F, indices) 
        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):
            break
        i=i+1 # Expand the size of the pseduo cluster set by 1
    return PeakIndices,Pfitness,C_Indices

def NeighborSearch(DisC, data, sample, P_indice, marked, radius):
    Cluster = []
    for i in range(np.shape(sample)[0]):
        if i not in marked:
            Dist = DisC[i, P_indice]
            if Dist <= radius:
                Cluster.append(i)         
    Indices = Cluster
    return Indices

def Sharing(fitness, indices):
    newfitness = fitness
    sum1 = 0
    for j in range(len(indices)):
        sum1 = sum1 + fitness[indices[j]]
    for th in range(len(indices)):
            newfitness[indices[th]] = fitness[indices[th]] / (1+sum1)    
    return newfitness
    
def Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    # Initialize the indices of Historical Pseduo Clusters and their fitness values
    HistCluster = PeakIndices
    HistClusterF = PseDuoF
    while True:
        # Call the merge function in each iteration
        [Cluster,Cfitness,F_Indices] = Pseduo_Merge(DisC, HistCluster, HistClusterF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma)
        # Check for the stablization of clutser evolution and exit the loop
        if len(np.unique(Cluster)) == len(np.unique(HistCluster)):
            break
        # Update the feature indices of historical pseduo feature clusters and
        # their corresponding fitness values
        HistCluster=Cluster
        HistClusterF=Cfitness
        C_Indices = F_Indices
    # Compute final evolved feature cluster information 
    FCluster = np.unique(Cluster)
    Ffitness = Cfitness
    C_Indices = F_Indices
    return FCluster, Ffitness, C_Indices
#----------------------------------------------------------------------------------------------------------
def Pseduo_Merge(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    # Initialize the pseduo feature clusters lables for all features 
    F_Indices = C_Indices
    # Initialize the temporal sample space for feature means and stds
    sample = np.vstack((DC_Mean,DC_Std)).T
    ML = [] # Initialize the merge list as empty
    marked = [] #List of checked Pseduo Clusters Indices 
    Unmarked = [] # List of unmerged Pseduo Clusters Indices 
    for i in range(len(PeakIndices)):
            M = 1 # Set the merge flag as default zero
            MinDist = math.inf # Set the default Minimum distance between two feature clusters as infinite
            MinIndice = 0 # Set the default Neighboring feature cluster indices as zero
            # Check the current Pseduo Feature Cluster has been evaluated or not
            if PeakIndices[i] not in marked:
                for j in range(len(PeakIndices)):
                        if j != i:
                            # Divergence Calculation between two pseduo feature clusters
                            D = DisC[PeakIndices[i], PeakIndices[j]]
                            if MinDist > D:
                                MinDist = D
                                MinIndice = j
                if MinIndice != 0:
                    # Current feature pseduo cluster under check
                    Current = sample[PeakIndices[i],:]
                    CurrentFit = PseDuoF[i]
                    # Neighboring feature pseduo cluster of the current checked cluster
                    Neighbor = sample[PeakIndices[MinIndice],:]
                    NeighborFit = PseDuoF[MinIndice]
                    
                    # A function to identify the bounady feature instance between two 
                    # neighboring pseduo feature clusters
                    BP=Boundary_Points(DisC, F_Indices,data, PeakIndices[i], PeakIndices[MinIndice])
                    BPF=fitness[BP]
                    if BPF<0.85*min(CurrentFit,NeighborFit):
                        M=0 # Change the Merge flag
                    if M == 1:
                        ML.append([PeakIndices[i],PeakIndices[MinIndice]])
                        marked.append(PeakIndices[i])
                        marked.append(PeakIndices[MinIndice])
                    else:
                        Unmarked.append(PeakIndices[i])
    NewPI = []
    # Update the pseduo feature clusters list with the obtained mergelist 
    for m in range(np.shape(ML)[0]):
        # print(ML[m][0],ML[m][1])
        if fitness[ML[m][0]] > fitness[ML[m][1]]:
            NewPI.append(ML[m][0])
            F_Indices[C_Indices==ML[m][1]] = ML[m][0]
        else:
            NewPI.append(ML[m][1])
            F_Indices[C_Indices==ML[m][0]] = ML[m][1]
    # Update the pseduo feature clusters list with pseduo clusters that have not appeared in the merge list 
    for n in range(len(PeakIndices)):
        if PeakIndices[n] in Unmarked:
            NewPI.append(PeakIndices[n])
    # Updated pseduo feature clusters information after merging
    FCluster = np.unique(NewPI)
    Ffitness = fitness[FCluster]
    return FCluster, Ffitness, F_Indices

def Boundary_Points(DisC, F_Indices, data, Current, Neighbor):
    [N, dim] = np.shape(data)
    TempCluster1 = np.where(F_Indices == Current)
    TempCluster2 = np.where(F_Indices == Neighbor)
    TempCluster = np.append(TempCluster1,TempCluster2)
    D = []
    for i in range(len(TempCluster)):
        D1 = DisC[TempCluster[i], Current]
        D2 = DisC[TempCluster[i], Neighbor]
        D.append(abs(D1 - D2))
    FI = np.argmin(D)
    BD = TempCluster[FI]
    return BD

def PseduoGeneration(PseP,N):
    Pse_Mean = PseP[:,0]
    Pse_Std = PseP[:,1]
    Data = np.zeros((N,len(Pse_Mean)))
    for i in range(len(Pse_Mean)):
        Data[:, i] = (np.repeat(Pse_Mean[i],N) + Pse_Std[i] * np.random.randn(N)).T
    return Data

def Psefitness_cal( PseP, sample, data, PseduoData, StdF, gamma):
    OriFN = np.shape(sample)[0]
    PN = np.shape(PseP)[0]
    PsePF = np.zeros(PN)
    for i in range(PN):
        TempSum = 0
        for j in range(OriFN):
            Var1 = np.var(data[:,j])
            Var2 = np.var(PseduoData[:,i])
            P = np.corrcoef(data[:,j],PseduoData[:,i])[0,1]
            Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
            D_KL = Sim / (Var1 + Var2)
            TempSum = TempSum + (math.exp(-(D_KL**2)/StdF))**gamma
        PsePF[i] = TempSum
    return PsePF
    
#--------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    [data,label] = Input()   
    [N, dim] = np.shape(data)
    [DC_means, DC_std] = Distribution_Est(data,dim)  
    [DisC,Dist] =  Feature_Dist(data,dim)
    StdF = max(Dist)
    end1 = time.time()
    print('The Distance Calculation time in seconds:',start-end1)
    gamma = 5
    fitness = fitness_cal(DisC, DC_means, DC_std, data, StdF, gamma)
    oldfitness = np.copy(fitness)
    [PeakIndices,Pfitness,C_Indices] = Pseduo_Peaks(DisC, Dist, DC_means,DC_std,data,fitness,StdF,gamma)
    fitness = oldfitness
    # Pseduo Clusters Infomormation Extraction
    PseDuo = DC_means[PeakIndices] # Pseduo Feature Cluster centers
    PseDuoF = Pfitness # Pseduo Feature Clusters fitness values
    #-------------Check for possible merges among pseduo clusters-----------#
    [FCluster,Ffitness,C_Indices] = Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, 
                                                  DC_means, DC_std, data, fitness, StdF, gamma)
    SF = FCluster
    Extract_FIndices = SF
    label = label.reshape(N,1)
    Extract_Data = np.concatenate((data[:,SF],label),axis=1)
    end = time.time()
    print('The total time in seconds:',end-start)
