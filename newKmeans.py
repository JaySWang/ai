# -*- coding: utf-8 -*-
import math
from numpy import *
#C:\\\\Users\\\\MrLevo\\\\Desktop\\\\machine_learning_in_action\\\\Ch10\\\\testSet.txt

#载入数据，清洗数据保存为矩阵形式
def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    dataMat = []
    for line in lines:
        result = line.strip().split('\t')
        fltline = list(map(float,result))
        dataMat.append(fltline)
    return dataMat


#向量计算距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


# 给定数据集构建一个包含k个随机质心的集合，
def randCent(dataSet,k):
    n = shape(dataSet)[1] # 计算列数

    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) #取每列最小值
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1) # random.rand(k,1)构建k行一列，每行代表二维的质心坐标
        #random.rand(2,1)#产生两行一列0~1随机数
    return centroids

#minJ + rangeJ*random.rand(k,1)自动扩充阵进行匹配，实现不同维数矩阵相加,列需相同


#一切都是对象
def kMeans(dataSet,k,distMeas = distEclud,creatCent = randCent):
    m = shape(dataSet)[0] # 行数
    clusterAssment = mat(zeros((m,2))) # 建立簇分配结果矩阵，第一列存索引，第二列存误差
    centroids = creatCent(dataSet,k) #聚类点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf # 无穷大
            minIndex = -1 #初始化
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:]) # 计算各点与新的聚类中心的距离
                if distJI < minDist: # 存储最小值，存储最小值所在位置
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):

            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A== cent)[0]]
            # nonzeros(a==k)返回数组a中值不为k的元素的下标
            #print type(ptsInClust)
            '''
            #上式理解不了可见下面的，效果一样
            #方法二把同一类点抓出来

            ptsInClust=[]
            for j in range(m):
                if clusterAssment[j,0]==cent:
                    ptsInClust.append(dataSet[j].tolist()[0])
            ptsInClust = mat(ptsInClust)
            #tolist  http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html
            '''

            centroids[cent,:] = mean(ptsInClust,axis=0) # 沿矩阵列方向进行均值计算,重新计算质心
    return centroids,clusterAssment

# 构建二分k-均值聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) # 初始化，簇点都为0
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 起始第一个聚类点，即所有点的质心

    centList =[centroid0] # 质心存在一个列表中

    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
        # 计算各点与簇的距离，均方误差，大家都为簇0的群

    while (len(centList) < k):

        lowestSSE = inf
        for i in range(len(centList)):

            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            # 找出归为一类簇的点的集合，之后再进行二分，在其中的簇的群下再划分簇
            #第一次循环时，i=0，相当于，一整个数据集都是属于0簇，取了全部的dataSet数据

            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #开始正常的一次二分簇点
            #splitClustAss，类似于[0   2.3243]之类的，第一列是簇类，第二列是簇内点到簇点的误差

            sseSplit = sum(splitClustAss[:,1]) # 再分后的误差和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 没分之前的误差
            print ("sseSplit: ",sseSplit)
            print ("sseNotSplit: ",sseNotSplit)
            #至于第一次运行为什么出现seeNoSplit=0的情况，因为nonzero(clusterAssment[:,0].A!=i)[0]不存在，第一次的时候都属于编号为0的簇

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                # copy用法http://www.cnblogs.com/BeginMan/p/3197649.html

        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #至于nonzero(bestClustAss[:,0].A == 1)[0]其中的==1这簇点，由kMeans产生

        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids


        centList.append(bestNewCents[1,:].tolist()[0])

        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE

    return mat(centList), clusterAssment


# dataMat =mat(loadDataSet('data/testSet2.txt'))
# myCentroids,clustAssing = kMeans(dataMat,4)
# print(myCentroids)
# print(clustAssing)


