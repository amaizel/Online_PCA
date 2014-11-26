def direction(N):
    
    B  = np.random.randn(2,N)

    B[1] = B[1]/np.linalg.norm(B[1])
    B[0] = B[0]/np.linalg.norm(B[0])
    A = np.dot( B[1] ,B[0] )

    B[0] = B[0] - A*B[1]

    B[1] = B[1]/np.linalg.norm(B[1])
    B[0] = B[0]/np.linalg.norm(B[0])
  
    return(B)

####################################
####################################
def setcorrel(N,D):
    
        
    C = np.identity(N)

    Dim1 = 10*np.dot(np.transpose(np.matrix(D[1])),np.matrix(D[1]))
    Dim2 =  25*np.dot(np.transpose(np.matrix(D[0])),np.matrix(D[0]))

    C = C + Dim1 + Dim2
    #C = np.cov(C)
    return(C)

######################################
######################################
def plotdata(A,t,N,cor,sigma):
   
   x = t
   y = A
   a = []
   b = []
   i = cor
   k = i
   j = 0
   while j < len(y):
         a.append(y[j]-sigma[j])
         b.append(y[j]+sigma[j])
         x[j]=x[j]/float(N) 
         j+=1
   if cor == 1:
      cor = 'red'
      letra = 'R'
   if cor == 2:
      cor = 'blue'
      letra = 'R'
   if cor == 3:
      cor = 'green'
      letra = 'Q'
      k = 2
      i = 1
   #fig, ax = plt.subplots()

#   while j < len(x): 
   plt.plot(x,y,'-', label= r'$%s_{%d%d}$'%(letra,i,k), color = cor)
   plt.legend()
   plt.xlabel(r'$\alpha$')
   plt.ylabel(r'$R(\alpha)$')
   plt.fill_between(x,a,b,alpha=0.4,color=cor) 
#     j+=1

   return

#######################################
#######################################
def projectdata(X,J,t):

   i = 0
   A = []
   while i<t:
         B = []
         projection = np.dot(X[i],J[0])
         B.append(projection)
         projection = np.dot(X[i],J[1])
         B.append(projection)
         A.append(B)
         i+=1

   return A
########################################
########################################
def setdist(N,t,B):

   mean = (0,)*N
 #  mean = np.random.rand(N,1)
   C = setcorrel(N,B)
   X = np.random.multivariate_normal(mean,C,t)
   return X

#########################################
#########################################
def  normalize(A):
  
     j = 0
     
     NORM = np.linalg.norm(A)

     while j < len(A): 
        A[j] = A[j]*(1./(NORM))
        j+=1
     return(A)

#########################################
#########################################
def soma(A,X,i,N):
    j = 0
    soma = np.zeros(N)
    while j<=i:
       soma = soma + np.dot(A[j],X)*A[j]

       j+=1
    return soma

##########################################
##########################################

import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

tmax = 2500
N= 250
t=1
passo = 100
samples = 0
nsamples = 100
eta1 = .1
eta2 = .09


overlap1 = []
overlap2 = []
q = []
sigmaq = []
data2 = []
data3 =[]
sigma1 = []
sigma2 = []
fig, ax = plt.subplots()
direc = direction(N)
#X = setdist(N,tmax,direc)

while t < tmax:
   samples = 0 
   data1 = []
   data2 = []
   data3=[]
   while samples < nsamples:


      X = setdist(N,t,direc)
      J = np.random.randn(2,N)

      J[1] = J[1]/np.linalg.norm(J[1])
      J[0] = J[0]/np.linalg.norm(J[0])
      i=0
      while i<t:
         A = (eta2/float(N))*np.dot(J[1],X[i])
         C = (eta1/float(N))*np.dot(J[0],X[i])
         SOMA1 = soma(J,X[i],1,N)
         SOMA2 = soma(J,X[i],0,N) 

         J[1] += A*(X[i] - SOMA1)
         J[0] += C*(X[i] - SOMA2) 

         J[1] = J[1]/np.linalg.norm(J[1])
         J[0] = J[0]/np.linalg.norm(J[0])
         i+=1

      data1.append(abs(np.dot(direc[0],J[0],)))
      data2.append(abs(np.dot(direc[1],J[1])))
      data3.append(abs(np.dot(J[0],J[1])))
      samples+=1
   
   q.append(np.mean(data3))
   sigmaq.append(np.std(data3))
   overlap1.append(np.mean(data1,dtype=np.float64))
   sigma1.append(np.std(data1,dtype=np.float64))
   overlap2.append(np.mean(data2, dtype=np.float64))
   sigma2.append(np.std(data2, dtype=np.float64))
   print(t,overlap1[t/passo],sigma1[t/passo],sigma2[t/passo])
    
   t+= passo


plotdata(overlap1,range(0,tmax-1,passo),N,1,sigma1)
plotdata(overlap2,range(0,tmax-1,passo),N,2,sigma2)
plotdata(q,range(0,tmax-1,passo),N,3,sigmaq)
    
plt.show() 

#plotdata(newdata,t)
