#spectrum of random function
from pylab import *
x=rand(128)
X=fft(x)
y=ifft(X)
c_[x,y]
print (abs((x-y).max()))

#spectrum of sin(5x)
x=linspace(0,2*pi,128)
y=sin(5*x)
Y=fft(y)
figure()
subplot(2,1,1)
plot(abs(Y),lw=2)
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5x)$")
grid(True)
subplot(2,1,2)
plot(unwrap(angle(Y)),lw=2)
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
show()

#improved spectrum of sin(5x)
x=linspace(0,2*pi,129)
x=x[:-1] #we create an array of 128 points, stopping just before 2pi
y=sin(5*x)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Improved spectrum of $\sin(5x)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$k$",size=16)
grid(True)
show()

#improved spectrum of (1+0.1cost)cos10t
t=linspace(-4*pi,4*pi,513)
t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/512.0
w=linspace(-64,64,513);w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

#2
def spectrum1(N,k,str):
	t=linspace(-6*pi,6*pi,N+1)
	t=t[:-1]
	if k==1:
		y=pow(sin(t),3)
	if k==2:
		y=pow(cos(t),3)
	Y=fftshift(fft(y))/N
	w=linspace(-64,64,N+1)
	w=w[:-1]
	figure()
	subplot(2,1,1)
	plot(w,abs(Y),lw=2)
	ylabel(r"$|Y|$",size=16)
	title(str)
	xlim([-15,15])
	grid(True)
	subplot(2,1,2)
	plot(w,angle(Y),'ro',lw=2)
	ylabel(r"Phase of $Y$",size=16)
	xlabel(r"$\omega$",size=16)
	xlim([-15,15])
	grid(True)
	show()
spectrum1(512,1,r"Spectrum of $sin^3(t)$")
spectrum1(512,2,r"Spectrum of $cos^3(t)$")

#3
N=1024
def spectrum2(N,k,str):
	t=linspace(-10*pi,10*pi,N+1)
	t=t[:-1]
	if k==1:
		y=cos(20*t+5*cos(t))
	if k==0:
		y=exp(-(t*t)/2)
	Y=fftshift(fft(y))/N
	w=linspace(-1,1,N+1)

	w=w[:-1]
	figure()
	subplot(2,1,1)
	plot(w,abs(Y),lw=2)
	ylabel(r"$|Y|$",size=16)
	title(str)
	grid(True)
	subplot(2,1,2)
	ii=where(abs(Y)>(k*1e-3))
	plot(w[ii],angle(Y[ii]),'ro',lw=2)
	ylabel(r"Phase of $Y$",size=16)
	xlabel(r"$\omega$",size=16)
	grid(True)
	show()

spectrum2(1024,1,r"Spectrum of $cos(20t+5cos(t))$")
spectrum2(128,0,r"Spectrum of $exp(-t^2/2)$")


