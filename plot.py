import numpy as np


import matplotlib.pyplot as plt

xAxis=[]
yAxis=[]

with open('advectionHeun.txt', 'r') as file:    
    lines = file.readlines()
    for line in lines:
        values = line.strip().split(',')
        print('{}   {}\n'.format(values[0], values[1]))
        xAxis.append(float(values[0]))
        yAxis.append(float(values[1]))

plt.plot(xAxis, yAxis,'*')

plt.grid(True)
plt.ylabel('Max Value')
plt.xlabel('t')
plt.autoscale(True)
plt.show()