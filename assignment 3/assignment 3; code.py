'''
----------------------------------------------
Assignment 3: EE2703 (Jan-May 2020)
Sarvani Cheruvu, EE18B120
----------------------------------------------
'''

from pylab import *
import scipy.special as sp
import scipy
import numpy as np

#2: loading data
yf=np.loadtxt("fitting.dat")
(N,k)=yf.shape
t=yf[:,0]

#3: all columns
#standard deviation
scl=logspace(-1,-3,9)	
figure(0)
for i in range(1,k):
	plot(t,yf[:,i],label='stddev={}'.format(around(scl[i-1]),5))
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Plot of the data to be fitted')

#4 : true value
def g(tk=t,A=1.05,B=-0.105):
	return A*sp.jn(2,tk)+B*t
y=g()
plot(t,y,label='true')
legend()
grid(True)
show()

#5: first column with error bars
figure(1)
plot(t,y,label='true')
errorbar(t[::5],yf[:,1][::5],scl[0],fmt='ro')
xlabel(r'$t$')
title(r'Data points along with error for standard deviation=0.10')
grid(True)
show()

#6: creating matrix equation
y1=sp.jn(2,t)
M=c_[y1,t]

#7: computing mean squared error 
n=21
A=linspace(0,2,n)
B=linspace(-0.2,0,n)
eps=np.zeros((n,n))
for i in range(n):
	for j in range(n):
		eps[i][j]=mean(square(yf[:,1]-g(t,A[i],B[j])))

#8: plotting contours
figure(2)
pl=contour(A,B,eps,levels=20)
xlabel(r'$A$')
ylabel(r'$B$')
title(r'Contour plot')
clabel(pl)
grid(True)
show()

#9: fitting data by least squares 
ex=np.zeros((2,1))
ex=scipy.linalg.lstsq(M,y)[0]

#10: error in the estimate
fit=np.zeros((k-1,2))
for i in range(k-1):
	fit[i]=scipy.linalg.lstsq(M,yf[:,i+1])[0]
Ae=np.zeros((k-1,1))
Be=np.zeros((k-1,1))
for i in range(k-1):
	Ae[i]=square(fit[i][0]-ex[0])
	Be[i]=square(fit[i][1]-ex[1])
figure(3)
plot(scl,Ae,label='A')
plot(scl,Be,label='B')
xlabel('Noise standard deviation')
ylabel('MS error')
title('Error in the estimate')
legend()
grid(True)
show()

#11: replotting using loglog
loglog(scl,Ae,'ro',label='Aerr(logscale)')
loglog(scl,Be,'go',label='Berr(logscale)')
errorbar(scl, Ae, std(Ae), fmt='ro')
errorbar(scl, Be, std(Be), fmt='go')
xlabel('Noise standard deviation')
ylabel('MS error (logscale)')
title('Error in the estimate (logscale)')
legend()
grid(True)
show()
