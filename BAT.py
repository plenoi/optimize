# -*- coding: utf-8 -*-
"""
Aj. Plenoi as CAMT CMU
Modified from
https://github.com/7ossam81/EvoloPy
https://www.scitepress.org/Papers/2016/60482/60482.pdf
"""
import numpy as np
import random
import time


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.population=[]
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

def BAT(objf,lb,ub,SearchAgents_no,Max_iteration):
    
    n=SearchAgents_no;      # Population size
    #lb=-50
    #ub=50
    dim = len(lb)
    N_gen=Max_iteration  # Number of generations
    
    A=0.5;      # Loudness  (constant or decreasing)
    r=0.5;      # Pulse rate (constant or decreasing)
    
    Qmin=0         # Frequency minimum
    Qmax=2         # Frequency maximum
    
    
    d=dim           # Number of dimensions 
    
    # Initializing arrays
    Q=np.zeros(n)  # Frequency
    v=np.zeros((n,d))  # Velocities
    Convergence_curve=[];
    
    # Initialize the population/solutions
    Sol = np.zeros((n,d))
    for i in range(dim):
      Sol[:, i] = np.random.randint(0,2,n) * (ub[i] - lb[i]) + lb[i]

    S=np.zeros((n,d))
    S=np.copy(Sol)
    Fitness=np.zeros(n)
    
    
    # initialize solution for the final results   
    s=solution()
    print("BAT is optimizing  \""+objf.__name__+"\"")    
    
    # Initialize timer for the experiment
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    #Evaluate initial random solutions
    for i in range(0,n):
      Fitness[i]=objf(Sol[i,:])
    
    
    # Find the initial best solution
    i=np.argmin(Fitness)
    best=np.copy(Sol[i,:])       
    fmin = Fitness[i]
    
    # Main loop
    for t in range (0,N_gen): 
        
        # Loop over all bats(solutions)
        for i in range (0,n):
          Q[i]=Qmin+(Qmin-Qmax)*random.random()
          v[i,:]=v[i,:]+(Sol[i,:]-best)*Q[i]
          S[i,:]=Sol[i,:]+v[i,:]
          
          # Check boundaries
          for j in range(d):
            Sol[i,j] = np.clip(Sol[i,j], lb[j], ub[j])
            

    
          # Pulse rate
          if random.random()>r:
              S[i,:]=best+0.001*np.random.randn(d)
          
          # Evaluate new solutions
          # Binary BAT by sigmoid
          for j in range(d):
              Xn=S[i,j]
              TF=1/(1+np.exp(-10*(Xn-0.5)));
              if TF >= np.random.uniform(0,1):
                  X_binary=1 
              else:
                  X_binary=0
              S[i,j]=X_binary
          
          Fnew=objf(S[i,:])
          
          # Update if the solution improves
          if ((Fnew<=Fitness[i]) and (random.random()<A) ):
                Sol[i,:]=np.copy(S[i,:])
                Fitness[i]=Fnew;
           
    
          # Update the current best solution
          if Fnew<=fmin:
                best=np.copy(S[i,:])
                fmin=Fnew
                
        #update convergence curve
        Convergence_curve.append(fmin)        

        if (t%1==0):
            print(['At iteration '+ str(t)+ ' the best '+str(numFold)+'CV accuracy is '+ str(1-fmin)]);
    
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="BAT"   
    s.bestIndividual = best
    s.population = Sol
    s.objfname=objf.__name__
    
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
s = BAT(fitness,lb,ub,40,20)
