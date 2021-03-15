# -*- coding: utf-8 -*-
import numpy as np
"""

@time    : 2021/3/14 14:47
@author  : Yuqiao Zhao
@contact : zhaoyuqiao97@gmail.com
@file    : svmMLiA.py
@software: PyCharm

"""
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4]),float(lineArr[5])])
        labelMat.append(float(lineArr[6]))
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, MaxIter):
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < MaxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)): #如果alpha可以更改进入优化过程
                j = selectJrand(i,m) #随机选择第二个alpha
                fXj = float(np.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]: #保证alpha在0,C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T -dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); continue
                alphas[i] += labelMat[j] * labelMat[j] * (alphaJold - alphas[j]) #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold * dataMatrix[i,:] * dataMatrix[j,:].T)
                b2 = b- Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold * dataMatrix[j,:] * dataMatrix[j,:].T)
                if 0 < alphas[i] and C > alphas[i]: b = b1
                elif 0 < alphas[j] and C > alphas[j]: b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged))
            if(alphaPairsChanged == 0): iter += 1
            else: iter = 0
            print("iteration number: %d" %iter)
        return b,alphas

def kernelTrans(X, A, kTup):  # 通过数据计算转换后的核函数
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性核函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # 高斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'laplace':  # 拉普拉斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
            K[j] = np.sqrt(K[j])
        K = np.exp(-K / kTup[1])
    elif kTup[0] == 'poly':  # 多项式核
        K = X * A.T
        for j in range(m):
            K[j] = K[j] ** kTup[1]
    elif kTup[0] == 'sigmoid':  # Sigmoid核
        K = X * A.T
        for j in range(m):
            K[j] = np.tanh(kTup[1] * K[j] + kTup[2])
    else:
        raise NameError('执行过程出现问题 -- \
    核函数无法识别')
    return K



class optStruct: #完整版PlattSMO的支持函数
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k): #计算误差 E = f(xi) - yi
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei): #根据最大误差选取J
    maxK = -1                       #直观来看，KKT条件违背的程度越大，则更新变量更新后可能导致的目标函数增幅越大，所以SMO算法先选取违背KKT条件最大的变量，
    maxDeltaE = 0                   # 第二个变量的选择一个使目标函数值增长最快的变量。这里SMO采用了一种启发式，
    Ej = 0                          # 是选取的两个变量所对应的样本的间隔最大，一种直观的解释，这样的两个变量有很大的区别，与对两个相似变量更新相比，
    oS.eCache[i] = [1, Ei]          # 对他们更新会使目标函数值发生更大的变化.（参考西瓜书）
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    # print(validEcacheList)
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k): #更新误差
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):  #SMO算法
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0



#根据输入的数据集，标签，松弛因子C，容忍度toler，最大迭代数与核函数参数，运行完整的SMO算法，直到更新达到一定条件（达到迭代最大次数，更新程度达到一定变化范围），停止迭代，返回参数alpha，b.
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('rbf', 1.3)):  #核函数选择并计算参数alpha，b
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calculateW(alphas,dataArr,labelArr):    #计算系数W
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataArr), np.array(labelArr)
    sum = 0
    for i in range(np.shape(dataMat)[0]):
        sum += alphas[i]*labelMat[i]*dataMat[i].T
    print(sum)
    return sum


def testRbf(fileTrain,fileTest,k1=1.3):
    dataArr, labelArr = loadDataSet(fileTrain)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    print("alphas:",alphas)
    print("b:",b)
    dataArr, labelArr = loadDataSet(fileTest)
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

def divide_data(filename):
    fr = open(filename)
    lineArr = fr.readlines()
    with open('test1.data', 'w') as f1:
        with open('test2.data', 'w') as f2:
            with open('test3.data', 'w') as f3:
                with open('test4.data', 'w') as f4:
                    with open('test5.data', 'w') as f5:
                        for i in range(len(lineArr)):
                            if i % 5 == 0:
                                f1.write(lineArr[i])
                            elif i % 5 == 1:
                                f2.write(lineArr[i])
                            elif i % 5 == 2:
                                f3.write(lineArr[i])
                            elif i % 5 == 3:
                                f4.write(lineArr[i])
                            elif i % 5 == 4:
                                f5.write(lineArr[i])
    f5.close()
    f4.close()
    f3.close()
    f2.close()
    f1.close()


if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet("training.data")
    # smoSimple(dataArr,labelArr,0.6,0.001,40)
    divide_data("training.data")
    testRbf("training1.data","test1.data")
    testRbf("training2.data", "test2.data")
    testRbf("training3.data", "test3.data")
    testRbf("training4.data", "test4.data")
    testRbf("training5.data", "test5.data")