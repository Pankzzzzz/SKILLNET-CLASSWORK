import numpy as np
import matplotlib.pyplot as plt


bins =[15,20,25,30,40,45]
datascience_students_age =[23,32,43,25,42,33]
javascript_students_age = [21,42,54,23,24,24]
plt.hist(datascience_students_age,bins,width=0.2,histtype='bar',orientation='vertical',color ='blue',label='datascience_students_age')
plt.hist(javascript_students_age,bins,rwidth=0.2,histtype='bar',orientation='vertical',color ='red',label='javascript_students_age')
plt.title("DATASCIENCE STUDENTS AGE")
plt.xlabel('NO. of students age')
plt.legend(loc=10)
plt.show()
