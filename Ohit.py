import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import floor
import numpy as np

class Ohit:
    def __init__(self,X,y,Kn = None,c1 = 3,c2 = 2,c3 = 2,HDIC_Type='HDAIC',init = None,conf = 0.01,intercept = True):
        # -----------------------#
        # X is a n*p dataframe/numpy
        # y is a n*1 dataframe/numpy
        # init is index of column
        # -----------------------#
        if type(X).__module__ == 'numpy' or type(X) == list:
            X = pd.DataFrame(X)
            X.columns = ['V'+str(i+1) for i in range(X.shape[1])]
        if type(y).__module__ == 'numpy' or type(y) == list:    
            y = y.reshape(-1)
            y = pd.DataFrame({'y':y})
        
        n,p = X.shape
        self.n = n # sample size
        self.p = p # dimension of covariates
        self.Kn = Kn # number of iteration
        self.X = X # covariates
        self.y = y # response
        self.c1 = c1 # multiplier for Kn if Kn is none
        self.c2 = c2 # penalty for HDAIC
        self.c3 = c3 # penalty for HDHQ
        self.type = HDIC_Type # HDIC type
        self.init = init # the subset which user specifies
        self.init_len = 0 # init's length
        self.conf = conf # confident interval level
        self.intercept = intercept
        
        if init != None:
            if type(init) != list:
                init = [init]
            self.init_len = len(init)
            for i,v in enumerate(init):
                if type(v) == int:
                    init[i] = self.X.columns[v]
        
    def OGA(self):
        n = self.n
        p = self.p
        Kn = self.Kn
        X = self.X
        y = self.y
        c1 = self.c1
        init = self.init
        init_len = self.init_len
        if n!=len(y):
            print('the number of observations in y is not equal to the number of rows of x')
        if n==1:
            print('the sample size should be greater than 1')
        if Kn == None:
            K = max(1, min(floor(c1*math.sqrt(n/math.log(p))),p))
        else:
            if ((Kn<1) or (Kn>p)):
                print( 'Kn should between 1 and dimensions of variables')
            if (Kn-np.floor(Kn))!=0 :
                print( 'Kn should be a positive integer')
            K  = Kn 
        if K > p-init_len:
            K = p-init_len
        self.K = K

        # interation 1
        ## normalize
        dy = y-np.mean(y)
        dx = X.apply(lambda x: x-np.mean(x), axis = 1)
        self.dy = dy.copy()
        self.dx = dx.copy()
        ## init preprocessing
        Jhat  = [None for _ in range(K+init_len)]
        if init != None:
            XJhat = dx[init]
            XJhat = XJhat/((XJhat**2).apply(sum, axis = 0))**(1/2)
            init_fit = sm.OLS(endog = dy,exog = XJhat).fit()
            dy = init_fit.resid
            Jhat[:init_len] = init
        ## set and likelihood setting
        Sigma2hat  = np.zeros(K, dtype= float)
        xnorms = ((dx**2).apply(sum, axis = 0))**(1/2)
        aSSE = np.abs(np.dot(dy.T,dx).reshape(-1)/xnorms)
        for i in [i for i in Jhat if i != None ]:
            aSSE[i] = 0
        Jhat[init_len] = aSSE.idxmax()
        try :
            rq = dx[[Jhat[init_len]]] - XJhat.dot(XJhat.T).dot(dx[[Jhat[init_len]]])
            rq = rq/(np.sum(rq**2))**(1/2)
            XJhat = pd.concat([XJhat,rq],axis = 1)
        except:
            rq = dx[[Jhat[init_len]]]
            XJhat = rq/(np.sum(rq**2))**(1/2)
        u = dy - XJhat.iloc[:,[init_len]].dot(XJhat.iloc[:,[init_len]].T).dot(dy)
        Sigma2hat[0] = np.mean(u**2)
        if K>1:
            for k in range(1+init_len,K+init_len):
                aSSE = (abs(np.dot(u.T,dx).reshape(-1)/xnorms))
                aSSE[Jhat[:k]] = 0
                Jhat[k] = aSSE.idxmax()
                rq = dx[[Jhat[k]]] - XJhat.dot(XJhat.T).dot(dx[[Jhat[k]]])
                rq = (rq/(np.sum(rq**2))**(1/2))
                XJhat = pd.concat((XJhat,rq),axis = 1) 
                u = u - XJhat.iloc[:,[k]].dot(XJhat.iloc[:,[k]].T).dot(u)
                fit = sm.OLS(endog = dy,exog = XJhat).fit()
                uPath = fit.resid
                Sigma2hat[k-init_len] = np.mean(uPath**2)
        self.Jhat  = Jhat
        self.Sigma2hat = Sigma2hat
        
    def HDIC(self):
        HDIC_Type = self.type
        n = self.n
        p = self.p
        X = self.X
        y = self.y
        dx = self.dx
        dy = self.dy
        c2 = self.c2
        c3 = self.c3
        init_len = self.init_len
        Sigma2hat = self.Sigma2hat

        # check HDIC type
        if HDIC_Type !='HDAIC' and HDIC_Type !='HDBIC'and HDIC_Type!='HDHQ':
            print( 'HDIC_Type error')
        if HDIC_Type == 'HDAIC':
            omega_n = c2
        if HDIC_Type == 'HDBIC':
            omega_n = np.log(n)
        if HDIC_Type == 'HDHQ':
            omega_n = c3 * np.log(np.log(n))
        
        # get min hdic
        hdic = (n * np.log(Sigma2hat)) + (np.arange(self.K)) * omega_n * np.log(p)
        kn_hat = np.argmin(hdic)
        J_HDIC = (self.Jhat[:(kn_hat+1+init_len)])
        J_Trim = self.Jhat[:(kn_hat+1+init_len)]
        trim_pos = np.zeros(kn_hat+init_len, dtype= bool)
        trim_pos[:init_len] = True

        # get benchmark for trim
        fit = sm.OLS(endog = dy,exog = dx[J_HDIC]).fit()
        uHDIC = fit.resid
        benchmark = n*np.log(np.mean(uHDIC**2)) + (kn_hat ) * omega_n * np.log(p)
        if kn_hat>0:
            # start from 
            for l in range(init_len,kn_hat+init_len):
                JDrop1 = np.delete(J_HDIC,l)
                fit = sm.OLS(endog = dy,exog = dx[JDrop1]).fit()
                uDrop1 = fit.resid
                HDICDrop1 = n*np.log(np.mean(uDrop1**2)) + (kn_hat - 1) * omega_n * np.log(p)
                if HDICDrop1 >= benchmark:
                    trim_pos[l] = True
            if any(trim_pos[init_len:]):
                use_idx = np.where(trim_pos)[0]
                J_Trim =  [J_Trim[i] for i in use_idx]
            else :
                J_Trim = J_HDIC[:(init_len+1)]
        else :
            J_Trim = J_HDIC
        self.J_HDIC = list(np.sort(J_HDIC))
        self.J_Trim = list(np.sort(J_Trim))
    
    def predict(self,X_test):
        if X_test.shape[1] != self.X.shape[1]:
            print('new data has different columns with original data')
        try:
            self.J_Trim
        except:
            self.OGA()
            self.HDIC()
        if type(X_test).__module__ == 'numpy':
            X_test = pd.DataFrame(X_test)
            X_test.columns = ['V'+str(i+1) for i in range(X.shape[1])]
        if type(y_test).__module__ == 'numpy':    
            y_test = y_test.reshape(-1)
            y_test = pd.DataFrame(y_test)
        conf = self.conf
        X = self.X.copy()
        y = self.y.copy()
        n = self.n
        J_Trim = self.J_Trim.copy()
        # check intercept

        if self.intercept:
            allone = (X==1).all(axis = 0)
            if allone.any():
                intercept = int(np.where(allone)[0])
                if intercept not in J_Trim:
                    J_Trim.append(X.columns[intercept])
            else:
                X = pd.concat([X,pd.DataFrame({'intercept':[1 for _ in range(n)]})],axis = 1)
                X_test = pd.concat([X_test,pd.DataFrame({'intercept':[1 for _ in range(X_test.shape[0])]})],axis = 1)
                J_Trim.append('intercept')          

        # fit
        fit = sm.OLS(endog = y,exog = X[J_Trim]).fit()    
        self.fit = fit

        # predict
        self.yPred_train = fit.get_prediction(X[J_Trim]).summary_frame(alpha = conf)
        self.yPred_test = fit.get_prediction(X_test[J_Trim]).summary_frame(alpha = conf)
        

    def OGA_HDIC(self):
        self.OGA()
        self.HDIC()
        self.predict(self.X,self.y)
        self.yPred_test = None

