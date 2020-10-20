import scipy as sig
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
def f(w): 
	return sp.lti([1,w],np.polymul([1,2*w,w*w+1.5*1.5],[1,0,2.25]))

def plott(a,b,c,d,e):
	plt.title(a)
	plt.xlabel(b)
	plt.ylabel(c)
	plt.plot(d,e)
#1
t,x=sp.impulse(f(0.5),None,np.linspace(0,50,1000))

plott('Time response of spring with decay constant=0.5','t','x',t,x)
plt.show()
#2
t,x=sp.impulse(f(0.05),None,np.linspace(0,50,1000))
plott('Time response of spring with decay constant=0.05','t','x',t,x)
plt.show()
#3
t=np.linspace(0,100,5000)
k=np.linspace(1.4,1.6,5)
plt.title('Time response of spring with variable cosine arguments')
plt.xlabel('t')
plt.ylabel('x')
transfunction=sp.lti([1],[1,0,2.25])
for i in range(len(k)):
	t,y,svec=sp.lsim(transfunction,np.exp(-0.05*t)*np.cos(k[i]*t),t)
	plt.plot(t,y)
plt.legend(k)
plt.show()

#4
X=sp.lti([1,0,2],[1,0,3,0])
Y=sp.lti([2],[1,0,3,0])
t,x=sp.impulse(X,None,np.linspace(0,20,100))
t,y=sp.impulse(Y,None,np.linspace(0,20,100))
plott('Coupled spring problem: x','t','x',t,x)
plt.show()
plott('Coupled spring problem:y','t','x',t,y)
plt.show()

#5
R=100
L=1e-6
C=1e-6
twoport=sp.lti([1],[L*C,R*C,1])
w,S,phi=twoport.bode()
plott('Two port network: magnitude response','t','|H|',np.log(w),S)
plt.show()
plott('Two port network: phase response','t','<H',np.log(w),phi)
plt.show()

#6
def f(n,test,title):
	t=np.linspace(0,n,30000)
	vi=np.cos((1e3)*t)-np.cos((1e6)*t)
	t,vo,svec=sp.lsim(twoport,vi,t)
	plott(title,'t','vo',t,vo)
	plt.show()
f(30e-6,1,'Two port network: output for t < 30 microseconds')
f(10e-3,1,'Two port network: output for t < 10 milliseconds')
