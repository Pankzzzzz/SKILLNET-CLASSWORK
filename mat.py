# import matplotlib
# print(matplotlib.__version__)

# print("hello")
# import numpy as np
# import matplotlib.pyplot as plt 

# Diameter = 10
# radius = Diameter/2

# area = 3.14*radius*radius
# print(area)

# a= area
# b= area/2
# c= area/3
# d= area/4

# xpoint = np.array([a,b])*2
# ypoint = np.array([c,d])

# plt.plot(xpoint,ypoint)
# plt.show()

# x = np.array([1,2,54,34,6,232,53,55,452,352,523,53,5,3557,4656,34,324])
# y = np.array([3,8,64,53,34,43,465,564,4578,435,74,34,574,5654,3456,32,43])

# plt.plot(x,y)
# plt.show()

# x= np.array([0,0,0,0,0,0,0,0,0,0,0,0])


# plt.plot(x)
# plt.show()

# x = np.array([1,2,3,4,5])
# plt.plot(x,marker='X')
# plt.show()

# x = np.array([1,222,3,43,25,6])

# plt.plot(x,'o-.k',marker='D',ms=50)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# x= np.array([1,2,3,4,5])

# plt.plot(x,'o:r',linewidth=20,ms=20)
# plt.show()

# y1= np.array([3,9,2,11,43,23])
# y2= np.array([6,8,7,5,6,9])
# plt.plot(y1,'o:r')
# plt.plot(y2)
# plt.show()

# x1 = np.array([1,3,4,3,7,21])
# x2=np.array([1,3,5,7,8,9])
# font1={'family':'serif','color':'darkred','size':'20'}
# font2={'family':'serif','color':'blue','size':'35'}
# plt.title('Extra juicy',fontdict=font1,loc='left')
# plt.xlabel('sales',fontdict=font2)

# plt.grid(color='pink',ls='--',lw='20')
# plt.subplot(x1,x2)

# plt.show()

x= np.array([0,1,2,3])
y=np.array([2,4,1,10])

plt.subplot(2,1,1)
plt.plot(x,y,'o-.r',lw=5,color='blue')

x= np.array([0,1,2,3])
y=np.array([2,4,1,10])
plt.subplot(2,1,2)
plt.plot(x,y,'o-.r',lw=5,color='black')


x= np.array([0,1,2,3])
y=np.array([2,4,1,10])
plt.title('YOOOOOOO')

plt.subplot(2,2,1)
plt.plot(x,y,'o-.r',lw=5,color='blue')

x= np.array([0,1,2,3])
y=np.array([2,4,1,10])
plt.subplot(2,2,2)
plt.plot(x,y,'o-.r',lw=5,color='black')

plt.show()