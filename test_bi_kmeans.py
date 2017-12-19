# coding: utf-8
# kmeans/test_bi_kmeans.py

import kmeans
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataMat = np.mat(kmeans.loadDataSet('data/testSet2.txt'))
    k=4  

    testCount = 1;
    sseTotal = 0;
    for t in range(testCount):
        centroids, clusterAssment = kmeans.newBiKmeans(dataMat, k)
        sse = 0
        for i in range(len(centroids)):
             sse += sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
        print ("one sse",sse)   
        sseTotal+=sse
    print ("sse mean for biKmeans",sseTotal/testCount)   

    # sseTotal=0
    # for t in range(testCount):
    #     centroids, clusterAssment = kmeans.kMeans(dataMat, k)
    #     sse = 0
    #     for i in range(len(centroids)):
    #          sse += sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
    #     print ("one sse",sse)   
    #     sseTotal+=sse
    # print ("sse mean for kmeans",sseTotal/testCount)   


    clusterCount = centroids.shape[0]
    m = dataMat.shape[0]
    # 绘制散点图
    patterns = ['o', 'D','^','s']
    colors = ['b', 'g', 'black','m']
    fig = plt.figure()
    title = 'bi-kmeans with k='+str(k)
    ax = fig.add_subplot(111, title=title)
    print(clusterCount)
    for k in range(clusterCount):
        # 绘制聚类中心
        ax.scatter(centroids[k,0], centroids[k,1], color='r', marker='+', linewidth=20)
        for i in range(m):
            # 绘制属于该聚类中心的样本
            ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A==k)[0]]
            ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], marker=patterns[k], color=colors[k])
    plt.show()
