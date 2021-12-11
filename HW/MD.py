import math
import random
import numpy as np

#Multidimensional knapsack

#generates v,w: w is vector of weights
#in: pmin, pmax, eps
def generateInput(pmin,pmax,eps,d,wvar=.001):
    v = random.uniform(2*eps-wvar,2*eps+wvar) #long tailed distribution
    w = random.uniform(.0000001,eps)
    ws = []
    for dim in range(d):
        tr = True
        while tr:
            w = random.uniform(.0000001, eps)
            if pmax >= v/w >= pmin:
                ws.append(w)
                tr = False
    return v,ws


#input: w,v for time  t, beta*,
#output
def threshold(v,ws,phi):
    ret = 1
    for i in range(len(ws)):
        w = ws[i]
        if v/w <= phi[i]:
            ret = 0
    return ret

def part3main(dim=2):
    pmin = 2
    pmax = 300
    beta = 1 / (1 + math.log(pmax / pmin))

    # phi updates in loop
    phi = np.full((dim,1),beta)
    X = [] #decision variables
    y = np.full((dim,1),0)  # pocket utilizations
    obj = 0


    for t in range(5000):
        print("t" + str(t))
        v, w = generateInput(pmin, pmax,.001,dim)
        if (y < beta).any():
            xt = threshold(v, w, phi)
        else:
            #update phi vector
            phi = pmin * math.e ** (y / (beta) - 1)
            xt = threshold(v, w, phi)
        if xt == 1:
            y = y + np.asarray(w).reshape((dim,-1))

        obj = obj + xt * v
        X.append(xt)
    print("global objective: " + str(obj))
    print("utilization of each pocket")
    print(y)

