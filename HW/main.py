import math
import random
import numpy as np
import cvxpy #integer program solver
import matplotlib.pyplot as plt

import MD

#generates v,w
#in: pmin, pmax, eps
def generateInput(pmin,pmax,eps,wvar=.001):
    v = np.random.beta(.2,1,1).item() #long tailed distribution
    w = random.uniform(.0000001,eps)
    if pmax >= v/w >= pmin:
         return v,w
    else:
        return generateInput(pmin,pmax,eps)

#input: w,v for time  t, beta*,
#output
def threshold(v,w,phi):
    if v/w > phi:
        return 1
    else:
        return 0


#adapted from https://towardsdatascience.com/integer-programming-in-python-1cbdfa240df2
#in: weights, values, capacity
#out: solution vector (1) if item i chosen, value of objective
def offlineSolve(w,v,cap):
    # The variable we are solving for
    selection = cvxpy.Variable(len(w),boolean=True)

    #s.t. capacity
    c1 = sum(cvxpy.multiply(w,selection)) <= cap

    kp = cvxpy.Problem(cvxpy.Maximize(sum(cvxpy.multiply(v,selection))), [c1])

    kp.solve(solver=cvxpy.GLPK_MI)

    sol = selection.value
    sol = np.asarray(sol)
    maxobj = sum(np.multiply(sol,v))

    return sol,maxobj

def plotCurve(x,y,title = "",xlab="",ylab=""):
    N = len(x)
    plt.plot(x,y)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()

def part1():
    #generate different sequences
    nseq = 50
    CRs = []
    for i in range(nseq):
        items = []  # store items for ofline solver

        pmin = 2
        pmax = 40
        beta = 1 / (1 + math.log(pmax / pmin))

        # phi updates in loop
        phi = pmin
        X = []
        y = 0  # knapsack utilization
        obj = 0

        for t in range(5000):
            v, w = generateInput(pmin, pmax, .001)
            if y < beta:
                xt = threshold(v, w, phi)
            else:
                phi = pmin * math.e ** (y / (beta) - 1)
                xt = threshold(v, w, phi)
            if xt == 1:
                y = y + w
            obj = obj + xt * v
            X.append(xt)
            items.append((v, w))


        items = np.asarray(items)
        offlineSol, offlineMax = offlineSolve(items[:, 1], items[:, 0], 1)
        CRs.append(offlineMax/obj)

        printResults(obj,offlineMax,offlineSol)

    print("***** Average CR over "+str(nseq) +"runs *****")
    print(np.mean(np.asarray(CRs)))


def part2():
    # generate different sequences
    nseq = 50
    CRs = []
    for i in range(nseq):
        items = []  # store items for offline solver

        pmin = 2
        pmax = 40
        beta = 1 / (1 + math.log(pmax / pmin))

        # phi updates in loop
        phi = pmin
        X = []
        y = 0  # knapsack utilization
        obj = 0
        phis = []
        ys = []

        for t in range(200):
            v, w = generateInput(pmin, pmax, .2)
            if y < beta:
                xt = threshold(v, w, phi)
            else:
                phi = pmin * math.e ** (y / (beta) - 1)
                xt = threshold(v, w, phi)
            if xt == 1:
                y = y + w
            phis.append(phi)
            ys.append(y)
            obj = obj + xt * v
            X.append(xt)
            items.append((v, w))

        items = np.asarray(items)
        offlineSol, offlineMax = offlineSolve(items[:, 1], items[:, 0], 1)
        CRs.append(offlineMax/obj)

        #printResults(obj, offlineMax, offlineSol)

        #plot the phi curve
        #plotCurve(ys,phis)

    print("***** Average CR over " + str(nseq) + "runs *****")
    print(np.mean(np.asarray(CRs)))

def part2Mod():
    # generate different sequences
    nseq = 50
    CRs = []
    for i in range(nseq):
        items = []  # store items for offline solver

        pmin = 2
        pmax = 40
        beta = 1 / (1 + math.log(pmax / pmin))

        # phi updates in loop
        phi = pmin
        X = []
        y = 0  # knapsack utilization
        obj = 0
        phis = []
        ys = []

        #key change here is anneal
        anneal = .0002
        y_prev = 0

        for t in range(200):
            v, w = generateInput(pmin, pmax, .2)
            if y < beta:
                xt = threshold(v, w, phi)
            else:
                phi = pmin * math.e ** (y / (beta) - 1)
                xt = threshold(v, w, phi)
            if xt == 1:
                y = y_prev + w
                y_prev = y
            else:
                y = y + anneal
            phis.append(phi)
            ys.append(y)
            obj = obj + xt * v
            X.append(xt)
            items.append((v, w))

        #Results
        items = np.asarray(items)
        offlineSol, offlineMax = offlineSolve(items[:, 1], items[:, 0], 1)
        CRs.append( offlineMax/obj)

        #print Results
        #printResults(obj, offlineMax, offlineSol)


        #view the phi function
        #plotCurve(ys,phis)

    print("***** Average CR over " + str(nseq) + "runs *****")
    print(np.mean(np.asarray(CRs)))

def printResults(obj,offlineMax,offlineSol):
    # print Results
    print("Online solution")
    # print(np.asarray(X))
    print("objective")
    print(obj)
    print("offline solution")
    # print(offlineSol)
    print("objective")
    print(offlineMax)
    print(sum(offlineSol))
    print("CR")
    print( offlineMax/obj)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #part1()
    #part2()
    #part2Mod()
    MD.part3main(2)