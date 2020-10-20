'''
----------------------------------------------
Assignment 2: EE2703 (Jan-May 2020)
Sarvani Cheruvu, EE18B120
----------------------------------------------
'''

'''
----------------------------------------------
Assumptions: 
-all circuits with reactive elements have only AC sources.
-netlists for DC sources are written as Vx n1 n2 value
----------------------------------------------
'''

#import necessary modules

import sys
import numpy as np
import math
import cmath

#variable definitions

n=1;k=0
start=1;end=0
ckt=0;freqi=-1
xlist=[];vlist=[];ilist=[];vcvslist=[];vccslist=[];cccslist=[];ccvslist=[];nodelist={'GND':0}

#function to reverse list

def Reverse(lines): 
    return [ele for ele in reversed(lines)] 		

# classes for various components

class vsource:
    def __init__(self,no,node1,node2,value,phase):
        self.no=no
        self.node1=node1
        self.node2=node2
        self.value=float(value)
        self.phase=float(phase)

class isource:
    def __init__(self,no,node1,node2,value,phase):
        self.no=no
        self.node1=node1
        self.node2=node2
        self.value=float(value)
        self.phase=float(phase)

class impedance:
    def __init__(self,name,no,node1,node2,value):
        self.name=name
        self.no=no
        self.node1=node1
        self.node2=node2
        self.value=float(value)

class vcvs:
    def __init__(self,name,no,node1,node2,node3,node4,value):
        self.name=name
        self.no=no
        self.node1=node1
        self.node2=node2
        self.node3=node3
        self.node4=node4
        self.value=float(value)
class vccs:
     def __init__(self,name,no,node1,node2,node3,node4,value):
        self.name=name
        self.no=no
        self.node1=node1
        self.node2=node2
        self.node3=node3
        self.node4=node4
        self.value=float(value)
class ccvs:
    def __init__(self,name,no,node1,node2,voltage,value):
        self.name=name
        self.no=no
        self.node1=node1
        self.node2=node2
        self.voltage=voltage
        self.value=float(value)
class cccs:
     def __init__(self,name,no,node1,node2,voltage,value):
        self.name=name
        self.no=no
        self.node1=node1
        self.node2=node2
        self.voltage=voltage
        self.value=float(value)

#main program
try:

    # taking input from file
    with open(sys.argv[1]) as f:
        lines=f.readlines()			
			   
        # truncating list to between .circuit and .end & reversing it
        for i in range(0,len(lines)):
            if '.circuit' in lines[i]:
                start=i
            if '.end' in lines[i]:
                end=i
            if '.ac' in lines[i]:
                freqi=i
                freq=2*math.pi*float(lines[freqi].split(' ')[2])
    
        lines=lines[start+1:end:1]

        if(start>end):
            print("Incorrect file format. Please check.")


        for i in range(0,len(lines)):

            st=lines[i].replace("\n","")
            if st.find("#")!=-1:
                st=st[0:st.index("#")].strip()
            list1=st.split(' ')
            
            #creating a dictionary for the list of nodes
            if list1[1] not in nodelist.keys():
                nodelist.update({list1[1]:n})
                n=n+1
                    
            if list1[2] not in nodelist.keys():
                nodelist.update({list1[2]:n})
                n=n+1

            #updating a list of components with necessary data
            if list1[0][0]=='V':
                if freqi==-1:
                    vlist.append(vsource(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[3],'0'))
                else:
                    ckt=1
                    vlist.append(vsource(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[4],list1[5]))

            if list1[0][0]=='R' or list1[0][0]=='C' or list1[0][0]=='L':
                xlist.append(impedance(list1[0][0],list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[3]))

            if list1[0][0]=='I':
                if ckt==0: 
                    ilist.append(vsource(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[3],'0'))
                elif ckt==1:                    
                    ilist.append(isource(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[4],list1[5]))

            if list1[0][0]=='E':
                if list1[3] not in nodelist.keys():
                    nodelist.update({list1[3]:n})
                    n=n+1
                    
                if list1[4] not in nodelist.keys():
                    nodelist.update({list1[4]:n})
                    n=n+1
                vcvslist.append(vcvs(list1[0][0],list1[0][1],nodelist[list1[1]],nodelist[list1[2]],nodelist[list1[3]],nodelist[list1[4]],list1[5]))
            
            if list1[0][0]=='G':
                if list1[3] not in nodelist.keys():
                    nodelist.update({list1[3]:n})
                    n=n+1
                    
                if list1[4] not in nodelist.keys():
                    nodelist.update({list1[4]:n})
                    n=n+1
                vccslist.append(vccs(list1[0][0],list1[0][1],nodelist[list1[1]],nodelist[list1[2]],nodelist[list1[3]],nodelist[list1[4]],list1[5]))

            if list1[0][0]=='H':
                ccvslist.append(ccvs(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[3],list1[4])) 

            if list1[0][0]=='F':
                cccslist.append(cccs(list1[0][1],nodelist[list1[1]],nodelist[list1[2]],list1[3],list1[4]))

        k=len(vlist)
        
        #defining arrays
        M=np.zeros((n+k,n+k),dtype=complex);b=np.zeros((n+k,1),dtype=complex)
        
        #filling M and b arrays
        for i in range(len(xlist)):
                x=int(xlist[i].node1)
                y=int(xlist[i].node2)
                if xlist[i].name=='R':
                    z=float(xlist[i].value)
                if xlist[i].name=='C':
                    z=1/(float(xlist[i].value)*1j*freq)
                if xlist[i].name=='L':
                    z=float(xlist[i].value)*1j*freq
                if x!=0:
                        
                        M[x,y]=M[x,y]-1/z
                        M[x,x]=M[x,x]+1/z
                if y!=0:
                        M[y,x]=M[y,x]-1/z
                        M[y,y]=M[y,y]+1/z

        for i in range(len(vlist)):
                x=int(vlist[i].node1)
                y=int(vlist[i].node2)
                if ckt==0:
                    z=float(vlist[i].value)
                if ckt==1:
                    t=float(vlist[i].phase)
                    z=float(vlist[i].value)*(cmath.cos(t)+cmath.sin(t)*1j)
                if y!=0:
                    M[y,n+i]=-1
                if x!=0:
                    M[x,n+i]=1
                M[n+i,y]=-1
                M[n+i,x]=1
                b[n+i,0]=z

        for i in range(len(ilist)):
            x=int(ilist[i].node1)
            y=int(ilist[i].node2)
            z=float(ilist[i].value)
            if x!=0:
                b[x,0]=-z
            if y!=0:
                b[y,0]=z

        M[0]=0
        for i in range(n+k):
            M[i,0]=0
        M[0,0]=1
        print("\nThe M matrix is:")
        print(M)
        print("\nThe b matrix is:")
        print(b)
        #solution
        x=np.linalg.solve(M,b)
        
        print("\nVoltage at corresponding nodes:")
        for i in range(n):
            if ckt==0:
                print('V',i,'=',x[i])
            if ckt==1:
                print('V',i,'=',x[i],'cos',freq,'t')
        print("\nCurrent through corresponding voltage sources:")
        for i in range(k):
            if ckt==0:
                print('I',i,'=',x[i+n])
            if ckt==1:
                print('I',i,'=',x[i+n],'cos',freq,'t')
                                                                                                 
except IOError:
    print("Please input proper file.")
    exit()




