#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:31:07 2020

@author: logistics
"""
############# Minimize #################
import random
import time
import numpy as np

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.population=[]
        
def cv(clf, X, y, nr_fold):
    ix = np.zeros(len(y))
    for i in range(0, len(y)):
        ix[i] = i
    ix = np.array(ix)
    
    allACC = np.zeros(nr_fold)
    allSENS = np.zeros(nr_fold)
    allSPEC = np.zeros(nr_fold)
    allMCC = np.zeros(nr_fold)
    #allAUC = np.zeros(nr_fold)
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)        
        #pr = clf.predict_proba(test_X)[:,1]   
        #p = np.round(pr)
        p = clf.predict(test_X)
        TP=0   
        FP=0
        TN=0
        FN=0
        for i in range(0,len(test_y)):
            if test_y[i]==0 and p[i]==0:
                TP+= 1
            elif test_y[i]==0 and p[i]==1:
                FN+= 1
            elif test_y[i]==1 and p[i]==0:
                FP+= 1
            elif test_y[i]==1 and p[i]==1:
                TN+= 1
        ACC = (TP+TN)/(TP+FP+TN+FN)
        SENS = TP/(TP+FN)
        SPEC = TN/(TN+FP)
        det = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if (det == 0):            
            MCC = 0                
        else:
            MCC = ((TP*TN)-(FP*FN))/det
        #AUC = roc_auc_score(test_y,pr)
        allACC[j] = ACC
        allSENS[j] = SENS
        allSPEC[j] = SPEC
        allMCC[j] = MCC
        #allAUC[j] = AUC
    #np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)
    return np.mean(allACC)

from sklearn.svm import SVC
def fitness(gene):
    gene = np.array(np.round(gene),dtype=int)
    f = np.where(gene[0:numFeat]==1)[0]
    ci = int(''.join(["%g"%item for item in gene[numFeat:numFeat+3]]),2)
    gi = int(''.join(["%g"%item for item in gene[numFeat+3:numFeat+numPar]]),2)
    c =parC[ci]
    g =parG[gi]
    clf = SVC(C=c,gamma=g) 
    X_train_norm = X[:,f]
    return 0.95*(1-cv(clf,X_train_norm,y ,numFold)) + 0.05*(len(f)/numFeat) 

def GWO(objf,lb,ub, SearchAgents_no,Max_iter):
    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30  
    #SearchAgents_no=5
    dim = len(lb)
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.randint(0,2, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        
    Convergence_curve=np.zeros(Max_iter)
    s=solution()

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Xn=(X1+X2+X3)/3 
                # Binary GWO by sigmoid
                TF=1/(1+np.exp(-10*(Xn-0.5)));
                if TF >= np.random.uniform(0,1):
                    X_binary=1 
                else:
                    X_binary=0
                
                Positions[i,j]=X_binary  # Equation (3.7)
                
        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best '+str(numFold)+'CV accuracy is '+ str(1-Alpha_score)]);
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="GWO"
    s.objfname=objf.__name__
    s.best = Alpha_score
    s.bestIndividual = Alpha_pos
    s.population = Positions

    return s

from sklearn.datasets import load_svmlight_file
data = load_svmlight_file("trainnorm.scl", zero_based=False)
X = data[0].toarray()
y = data[1]
parC = np.array([2 ** i for i in np.arange(0,8, dtype=float)])
parG = np.array([2 ** i for i in np.arange(-8,8, dtype=float)])

numPar = (3+4)
numFold = 10
numFeat = np.size(X,1)
lb = np.zeros(numFeat+numPar)
ub  = np.ones(numFeat+numPar)
s = GWO(fitness,lb,ub,40,20)


