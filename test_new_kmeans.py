# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import newKmeans as km
#注意导入自己的Kmeans的py文件

data3 = mat(km.loadDataSet('data/testSet2.txt'))
centList,myNewAssments =km.biKmeans(data3,3)


###################创建图表2####################

plt.figure(2) #创建图表2

ax3 = plt.subplot() # 图表2中创建子图1
plt.title("biK-means Scatter")
plt.xlabel('x')
plt.ylabel('y')

ax3.scatter(data3[:,0].tolist(),data3[:,1].tolist(),color='b',marker='o',s=100)
ax3.scatter(centList[:,0].tolist(),centList[:,1].tolist(),color='r',marker='o',s=200,label='Cluster & K=3')


#显示label位置的函数

ax3.legend(loc='upper right')
plt.show()