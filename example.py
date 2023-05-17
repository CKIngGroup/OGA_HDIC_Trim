import Ohit
from Ohit import Ohit
from sklearn.model_selection import train_test_split
from numpy import random
import numpy as np
def generate_data1(n,p,q):

    beta = [3,3.75,4.5,5.25,6,6.75,7.5,8.25,9,9.75]
    b = np.sqrt(3/(4*q))
    x_relevant = random.normal(0,1,size = (n,q))
    d = random.normal(0,0.5,size = (n,p-q))
    x_relevant_sum  = np.sum(x_relevant, axis = 1)
    x_irrelevant = np.apply_along_axis(lambda x:x+b*x_relevant_sum,0,d)
    epsilon = random.normal(0,1,size = (n))
    y = x_relevant @ beta + epsilon
    return np.concatenate([x_relevant,x_irrelevant],axis = 1),y,beta


if __name__ == '__main__':
    n = 400
    p = 4000
    q = 10
    X,y,_ = generate_data1(n,p,q)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = Ohit(X_train,y_train,HDIC_Type = 'HDBIC',intercept = False)
    model.OGA_HDIC()
    model.predict(X_test)
    print(model.J_Trim)
    print(model.J_HDIC)

    # if init
    model = Ohit(X_train,y_train,HDIC_Type = 'HDBIC',init = [500],intercept = False)
    model.OGA_HDIC()
    model.predict(X_test)
    print(model.J_Trim)
    print(model.J_HDIC)
    
# %%
