import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.optimize import curve_fit
import numpy as np
import sys
import math

Nx=25 
Ny=25 
radius=0.35
Niter=1500 
if(len(sys.argv)!=5):
	print("Insufficient number of arguments. Please enter 5.")
	exit()

Nx=int(sys.argv[1])
Ny=int(sys.argv[2])
radius=float(sys.argv[3])
while(radius>0.5):
	print("Please enter a value of radius less than 0.5 cm.")
	radius=float(input())
Niter=int(sys.argv[4])

phi=np.zeros((Ny,Nx))
y=np.linspace(-0.5,0.5,num=Ny)
x=np.linspace(-0.5,0.5,num=Nx)
Y,X=np.meshgrid(y,x)
ii=np.where(X*X+Y*Y<=radius*radius)
phi[ii]=1.0
plt.title('Contour plot of potential')
plt.xlabel('x')
plt.ylabel('y')
plt.contourf(Y,X,phi)
plt.plot(y[ii[1]],x[ii[0]],'ro')
plt.legend(["Electrode potential=1"])
plt.colorbar()
plt.show()

errors=np.zeros((Niter,1))
for k in range(Niter):
	oldphi=phi.copy()
	#update phi array
	phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
	#boundaries
	phi[1:-1,0]=phi[1:-1,1]
	phi[1:-1,Nx-1]=phi[1:-1,Ny-2]
	phi[0,1:-1]=phi[1,1:-1]
	phi[ii]=1
	#errors
	errors[k]=(abs(phi-oldphi)).max()

#plots

nk=np.transpose(np.arange(0,Niter,1))
plt.title('Loglog plot of errors')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.loglog(nk,errors)
plt.loglog(nk[::50],errors[::50],'ro')
plt.legend(['errors','every 50th point'])
plt.show()
plt.title('Loglog plot of errors: after 500 iterations')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.loglog(nk[500:],errors[500:])
plt.loglog(nk[500::50],errors[500::50],'ro')
plt.legend(['errors','every 50th point'])
plt.show()
plt.title('Semilogy plot of errors')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.semilogy(nk,errors)
plt.semilogy(nk[::50],errors[::50],'ro')
plt.legend(['errors','every 50th point'])
plt.show()
plt.title('Semilogy plot of errors: after 500 iterations')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.semilogy(nk[500:],errors[500:])
plt.semilogy(nk[500::50],errors[500::50],'ro')
plt.legend(['errors','every 50th point'])
plt.show()

#fit
def f(iter_var):
	pfit= np.polynomial.polynomial.polyfit(nk[iter_var::],np.log(errors[iter_var::]),1)
	return pfit
A1,B1=np.exp(f(0)[0]),f(0)[1]
A2,B2=np.exp(f(500)[0]),f(500)[1]
plt.title('Plot of errors along with linear fits')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.loglog(nk,errors)
plt.loglog(nk,A1*np.exp(B1*nk),linewidth=0.0,marker='o',color='r')
plt.loglog(nk[500::],A2*np.exp(B2*nk[500::]),linewidth=0.0,marker='o',color='g')
plt.legend(["Errors","Fit 1","Fit 2 (after 500 iterations)"])
plt.show()

#surface plot
fig1=plt.figure(4) # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
plt.title('The 3-D surface plot of the potential')
plt.xlabel('x')
plt.ylabel('y')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=plt.cm.jet,linewidth=2)
plt.show()
plt.title('Contour of potential, after computation')
plt.xlabel('x')
plt.ylabel('y')
plt.contourf(Y,X[::-1],phi)
plt.plot(y[ii[1]],x[ii[0]],'ro')
plt.show()
Jx=(phi[1:-1,0:-2]-phi[1:-1,2:])/2
Jy=(phi[:-2,1:-1]-phi[2:,1:-1])/2

#plotting current density
plt.title("Vector plot of current flow")
plt.xlabel('x')
plt.ylabel('y')
plt.quiver(Y[1:-1,1:-1],-X[1:-1,1:-1],-Jx[:,::-1],-Jy)
plt.plot(y[ii[1]],x[ii[0]],'ro')
plt.legend(["Electrode potential=1"])
plt.show()