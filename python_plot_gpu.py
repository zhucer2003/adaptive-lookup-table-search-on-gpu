import numpy as  np
import matplotlib.pyplot as plt

data=np.loadtxt('gpu.dat')
print data[:,2]/data[:,1]
plotfile='performance_gpu.pdf'
width = 0.1

rec=plt.bar(np.log10(data[:,0]),data[:,2]/data[:,1],width , color ='r')
plt.hold(True) 
rec1=plt.bar(np.log10(data[:,0])+width,data[:,4]/data[:,3],width, color = 'b' )
rec2=plt.bar(np.log10(data[:,0])+2*width,data[:,6]/data[:,5],width , color ='y')
plt.hold(True) 
rec3=plt.bar(np.log10(data[:,0])+3*width,data[:,8]/data[:,7],width, color = 'm' )
rec4=plt.bar(np.log10(data[:,0])+4*width,data[:,10]/data[:,9],width , color ='g')
plt.hold(True) 
rec5=plt.bar(np.log10(data[:,0])+5*width,data[:,12]/data[:,11],width, color = 'k' )
plt.xlabel("N")
plt.ylabel("speedup")
plt.legend((rec[0],rec1[0],rec2[0],rec3[0],rec4[0],rec5[0]),('cuda_512','opencl_512', 'cuda_1024','opencl_1024','cuda_256','opencl_256'))
#plt.show()
plt.savefig(plotfile)


