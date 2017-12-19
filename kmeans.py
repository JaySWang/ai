# coding: utf-8
# kmeans/kmeans.py
import numpy as np
from numpy import *

def loadDataSet(filename):
    """
    读取数据集
    Args:
        filename: 文件名
    Returns:
        dataMat: 数据样本矩阵
    """
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 通过map函数批量转换
        fitLine = list(map(float, curLine))
        dataMat.append(fitLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    计算两向量的欧氏距离
    Args:
        vecA: 向量A
        vecB: 向量B
    Returns:
        欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    """
    随机生成k个聚类中心
    Args:
        dataSet: 数据集
        k: 簇数目
    Returns:
        centroids: 聚类中心矩阵
    """
    _, n = dataSet.shape
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # 随机聚类中心落在数据集的边界之内
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, maxIter = 5):
    """
    K-Means
    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    """
    # 随机初始化聚类中心
    centroids = randCent(dataSet, k)
    m, n = np.shape(dataSet)
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 标识聚类中心是否仍在改变
    clusterChanged = True
    # 直至聚类中心不再变化
    iterCount = 0
    while clusterChanged and iterCount < maxIter:
        iterCount += 1
        clusterChanged = False
        # 分配样本到簇
        for i in range(m):
            # 计算第i个样本到各个聚类中心的距离
            minIndex = 0
            minDist = np.inf
            for j in range(k):
                dist = distEclud(dataSet[i, :],  centroids[j, :])
                if(dist < minDist):
                    minIndex = j
                    minDist = dist
            # 判断cluster是否改变
            if(clusterAssment[i, 0] != minIndex):
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # 刷新聚类中心: 移动聚类中心到所在簇的均值位置
        for cent in range(k):
            # 通过数组过滤获得簇中的点
            ptsInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A == cent)[0]]
            if ptsInCluster.shape[0] > 0:
                # 计算均值并移动
                centroids[cent, :] = np.mean(ptsInCluster, axis=0)
    # print("|||")              
    # print(clusterAssment)
    return centroids, clusterAssment

def biKmeans(dataSet, k):
    """
    二分kmeans算法
    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    """
    m, n = np.shape(dataSet)
    # 起始时，只有一个簇，该簇的聚类中心为所有样本的平均位置
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 设置一个列表保存当前的聚类中心
    currentCentroids = [centroid0]
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化点分配结果，默认将所有样本先分配到初始簇
    for j in range(m):
        clusterAssment[j, 1] = distEclud(dataSet[j, :], np.mat(centroid0))**2
    # 直到簇的数目达标
    while len(currentCentroids) < k:
        # 当前最小的代价
        lowestError = np.inf
        # 对于每一个簇
        for j in range(len(currentCentroids)):
            # 获得该簇的样本
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0], :]
            # 在该簇上进行2-means聚类
            # 注意，得到的centroids，其聚类编号含0，1

            centroids, clusterAss = kMeans(ptsInCluster, 2)
            # 获得划分后的误差之和
            splitedError = np.sum(clusterAss[:, 1])
            # 获得其他簇的样本
            ptsNoInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A != j)[0]]
            # 获得剩余数据集的误差
            nonSplitedError = np.sum(ptsNoInCluster[:, 1])
            # 比较，判断此次划分是否划算
            if (splitedError + nonSplitedError) < lowestError:
                # 如果划算，刷新总误差
                lowestError = splitedError + nonSplitedError
                # 记录当前的应当划分的簇
                needToSplit = j
                # 新获得的簇以及点分配结果
                newCentroids = centroids.A
                newClusterAss = clusterAss.copy()
        # 更新簇的分配结果
        # 第0簇应当修正为被划分的簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 0)[
            0], 0] = needToSplit
        # 第1簇应当修正为最新一簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 1)[
            0], 0] = len(currentCentroids)
        # print(needToSplit)
        # print(len(currentCentroids))
        # 被划分的簇需要更新
        currentCentroids[needToSplit] = newCentroids[0, :]
        # 加入新的划分后的簇
        currentCentroids.append(newCentroids[1, :])
        # 刷新点分配结果

        print("---------")
        # print(needToSplit)

        # print(newClusterAss)
        # print(clusterAssment)
        clusterAssment[np.nonzero(
            clusterAssment[:, 0].A == needToSplit
        )[0], :] = newClusterAss

    return np.mat(currentCentroids), clusterAssment


# 构建二分k-均值聚类
def newBiKmeans(dataSet, k, distMeas=distEclud):
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

            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2)
            #开始正常的一次二分簇点
            #splitClustAss，类似于[0   2.3243]之类的，第一列是簇类，第二列是簇内点到簇点的误差

            sseSplit = sum(splitClustAss[:,1]) # 再分后的误差和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 没分之前的误差
            # print ("sseSplit: ",sseSplit)
            # print ("sseNotSplit: ",sseNotSplit)
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

        # print ('the bestCentToSplit is: ',bestCentToSplit)
        # print ('the len of bestClustAss is: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids


        centList.append(bestNewCents[1,:].tolist()[0])

        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    
    return mat(centList), clusterAssment


