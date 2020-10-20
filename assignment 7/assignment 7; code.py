from sympy import *
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt 

s=symbols('s')

def lowpass(Vi=1,R1=1e4,R2=1e4,C1=1e-9,C2=1e-9,G=1.586):
	A=Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b=Matrix([0,0,0,-Vi/R1])
	V=A.inv()*b
	return V

def highpass(Vi=1,C1=1e-9,C2=1e-9,R1=1e4,R3=1e4,G=1.586):
	A=Matrix([[0,0,1,-1/G],[-(s*C2*R3)/(1+s*C2*R3),1,0,0],[0,G,-G,-1],[-(s*C1+s*C2+1/R1),s*C2,0,1/R1]])
	b=Matrix([0,0,0,-Vi*s*C1])
	V=A.inv()*b
	return V

def response(Vo,H):
	return Vo*H

def sympytolti(Y):
    Y = expand(simplify(Y))
    num_p,den_p = fraction(Y)
    num_p,den_p = Poly(num_p,s), Poly(den_p,s)
    num,den = num_p.all_coeffs(), den_p.all_coeffs()
    num,den = [float(f) for f in num], [float(f) for f in den]
    return sp.lti(num,den)


def magresponse(H,title):
	ww=np.logspace(0,8,801) #frequency
	ss=1j*ww #jw
	hf=lambdify(s,H,'numpy')
	v=hf(ss) #computes the sympy function at every jw
	plt.title(title)
	plt.loglog(ww,abs(v),lw=2)
	plt.grid(True)
	plt.show()	

def output(lti,input_v,t,title):
	t,vo,svec=sp.lsim(lti,input_v,t)
	plt.title(title)
	plt.plot(t,vo)
	plt.grid(True)
	plt.show()

Vl=lowpass()[3]
Vh=highpass()[3]
print(Vl,Vh)

Hl=sympytolti(Vl)
Hh=sympytolti(Vh)
print(Hl,Hh)

t=np.linspace(0,1e-3,10000)

#low pass filter
magresponse(Vl,'Magnitude of impulse response of a low pass filter')
output(Hl,(t>0),t,'Step response of a low pass filter')
output(Hl,((np.sin(2000*np.pi*t)+np.cos(2e6*np.pi*t)))*(t>0),t,'Response to a sum of sinusoids')

#high pass filter
magresponse(Vh,'Magnitude of impulse response of a high pass filter')
output(Hh,(t>0),t,'Step response of a high pass filter')
t=np.linspace(0,1e-2,1000)
output(Hh,(np.sin(1e7*t)*np.exp(-1e3*t))*(t>0),t,'Response to a high frequency damped sinusoid')
t=np.linspace(0,0.5,10000)
output(Hh,(np.sin(1e3*t)*np.exp(-1e1*t))*(t>0),t,'Response to a low frequency damped sinusoid')


'''
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo=V[3]
print(Vo)


#step response

t=np.linspace(0,1,10000)
H=sympytolti(response(Vo,1/(s)))
print(H)
t,vo,svec=sp.lsim(H,(np.sin(2000*3.14*t)+np.cos(2e6*3.14*t)),t)
plt.plot(t,vo)
plt.show()


A,b,V = highpass(1e-9,1e-9,1e4,1e4,1.586,1)
V=V[3]
t=np.linspace(0,1,10000)
print(V)
H=sympytolti(response(V,1/((s)*(s)+2.25)))
t,vo,svec=sp.lsim(H,(np.sin(3e4*t)*np.exp(-3e4*t)),t)
plt.plot(t,vo)
plt.show()


H=sympytolti(response(V,1/(s)))
'''
