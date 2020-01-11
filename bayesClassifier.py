import numpy as np
import pandas as pd
import math

class Bayes:

    df = np.zeros((2, 2))
    priors = []

    def __init__(self, dataset):
        
        self.data = dataset
        self.df = self.data.to_numpy()
        self.r, self.c = self.df.shape
        self.y = self.df[:, self.c-1]
        self.y = list(self.y)
        self.discrete = []
        self.continuous = [] 
        self.conditionalPdfs = []
        self.conditionalPmfs = []
        self.outputClasses = list(set(self.y))
        for clas in self.outputClasses:
            self.priors.append(self.y.count(clas)/len(self.y))
    
    def computePmf(self, column, i):

        self.discrete.append(i)
        column = list(column)
        labels = list(set(column))
        for label in labels:
            cP = [i, label]
            l = [i for i, x in enumerate(column) if x == label]   
            for clas in self.outputClasses:
                locs = [i for i, x in enumerate(self.y) if x == clas]
                indices = list(set(l).intersection(set(locs)))
                cP.append(len(indices)/self.y.count(clas))
            self.conditionalPmfs.append(cP)
    

    def computePdf(self, column, i):

        self.continuous.append(i)
        Ncolumn = list(column)
        for clas in self.outputClasses:
            pts = [i for i in Ncolumn if self.y[Ncolumn.index(i)] == clas]
            pts = np.asarray(pts)
            u = np.mean(pts)
            v = np.var(pts)
            self.conditionalPdfs.append([i, u, v])

    def determineType(self, column):

        c = list(column)
        d = list(set(c))
        t = [i for i in d if (i*10)%10 != 0]
        if t:
            return 1
        elif len(d) > 10:
            return 1
        else:
            return 0
    
    def naiveBayesClassifier(self):

        for i in range(0, self.c-1):
        
            col = self.df[:, i]
            typeFlag = self.determineType(col)
            if typeFlag == 0:
                self.computePmf(col, i)
            else:
                self.computePdf(col, i)

    def findCtP(self, i, x):
    
        p = []
        index = [self.conditionalPdfs.index(j) for j in self.conditionalPdfs if j[0] == i]
        for i in index:
            [_, u, v] = self.conditionalPdfs[i]
            if v == 0:
                v = 1
            p.append((1/math.sqrt(2*math.pi*v))*math.exp(-((x-u)**2)/(2*v)))
        return p
    
    def findCdP(self, i, l):
    
        a = [j for j in self.conditionalPmfs if (j[0] == i)]
        probArray = [j for j in a if (j[1] == l)][0]
        probArray = probArray[2:len(probArray)]
        return probArray
    
    def predict(self, test):

        size = len(test)
        PYX = []
        for i in range(0, size):
            if i in self.continuous:
                p = self.findCtP(i, test[i])
            else:
                p = self.findCdP(i, test[i])
            PYX.append(p)

        PYX = np.asarray(PYX)
        [r, c] = PYX.shape
        finalProb = []
        for i in range(0, c):
            p = 1
            for j in range(0, r):
                prod = PYX[j][i]*p
            finalProb.append(prod)
        return [PYX, finalProb]
      
def main():

    data = Bayes(pd.read_csv("C:\\Users\\Gautham\\Documents\\Projects\\data\\heart-disease-uci\\heart.csv"))
    
    data.naiveBayesClassifier()
    i = 0
    e = 0
    for test in data.df:
        yp = test[-1]
        test = list(test)
        test.pop()
        [pyx, p] = data.predict(test)
        i += 1
        if p.index(max(p)) != yp:
            e += 1
    print('Model Accuracy = ', (1- (e/i))*100)
    
if __name__ == '__main__':
    main()

