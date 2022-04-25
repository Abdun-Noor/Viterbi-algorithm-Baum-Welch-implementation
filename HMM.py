import numpy as np 
import math
from numpy.core.fromnumeric import var
from scipy.stats import norm  

def pdf(x, mean, var):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
    return (1 / (math.sqrt(2 * math.pi * var) )) * expo


def google_viterbi(y , A , mean , var):
    a = A.shape[0]
    V = [{}]
    
    for i in range(a):
        V[0][i] = {"prob": np.log( 1/a * norm.pdf(y[0] , mean[i] , var[i])), "prev": None}
    
    for t in range(1, len(y)):
        V.append({})
        for st in range(a):
            max_tr_prob = V[t - 1][0]["prob"] + np.log(A[0][st])
            prev_st_selected = 0
            for prev_st in range(1 , a):
                tr_prob = V[t - 1][prev_st]["prob"] + np.log( A[prev_st][st])
                
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob + np.log( norm.pdf(y[t] , mean[st] , var[st]) )
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}



    # for line in dptable(V):
    #     print(line)

    opt = []
    max_prob = -99999999999999
    best_st = None
        
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    print(V[len(V)-1][previous]["prev"])

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0 , V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

#    print(opt)
    return opt


def viterbi(y , A , mean , var):
    a = A.shape[0]
    path = { i:[] for i in range(a)}
    Tc = np.zeros(a)
    Tl = np.zeros(a)
    ip = np.full(a , 1/a)
    for i in range(a):
        Tc[i] =  ip[i] * pdf(y[0] , mean[i] , var[i]) 
    Tc[:] = Tc[:]/sum(Tc[:])
    for i in range(1 , len(y)):
        Tl = Tc
        Tc = np.zeros(a)
        for j in range(a):
            max_prob , max_state = max((( Tl[z] * A[z][j] * pdf(y[i] , mean[j], var[j])  , z )for z in range(a)))

            Tc[j] = max_prob
            path[j].append(max_state)
        Tc[:] = Tc[:]/sum(Tc[:])
    
    max_prob = - np.inf
    maxp = None
    for i in range(a):
        path[i].append(i)
        if(Tc[i]>max_prob):
            max_prob = Tc[i]
            maxp = path[i]    

#    print(Tc)
    return maxp

def forward( y , A , mean , var):
    N = A.shape[0]
    T = len(y)

    F = np.zeros((N,T))
    F[:,0] = [.25 , .75] * norm.pdf(y[0] , mean , var)
    F[:,0] = F[:,0]/sum(F[:,0])

    for t in range(1, T):
        for n in range(N):
            F[n,t] = np.sum([F[j,t-1] * A[j,n] * norm.pdf(y[t] , mean[n] , var[n] ) for j in range(N)]  )
        F[:,t] = F[:,t]/sum(F[:,t])

    return F

def backward(y , A , mean , var):
    N = A.shape[0]
    T = len(y)

    B = np.zeros((N,T))
    B[:,-1] = 1
    B[:,-1] = B[:,-1]/sum(B[:,-1])

    for t in  range(T-2 , -1 , -1) :
        for n in range(N):
            B[n,t] = np.sum([ B[j,t+1] * A[n,j] * norm.pdf(y[t+1] , mean[j] , var[j] ) for j in range(N) ])
        B[:,t] = B[:,t]/sum(B[:,t])

    return B

def baum_welch(y , A , mean , var):
    N = A.shape[0]
    T = len(y)

    forw = forward(y , A , mean , var)
    backw = backward(y , A , mean , var)

 #   print(forw)
 #   print(backw)

    prob =  np.sum(forw[:,-1])
    pp = forw * backw
    ss = np.sum(pp, axis = 0)
    gamma = pp/ss

    xi = np.zeros(( N, N , T-1))
    for i in range(xi.shape[2]):
        temp = (forw[:,i].reshape((N,1))*backw[:,i+1] ) * A * norm.pdf(y[i+1] , mean , var)
        temp = temp/np.sum(temp)
        xi[:,:,i] = temp 

    # for t in range(xi.shape[0]):
    #     xi[t,:,:] = A * forw[:,[t]] * norm.pdf(y[t+1], mean , var) * backw[:, t+1] / prob
    
    zi = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            zi[i,j] = np.sum([xi[i,j,k] for k in range(xi.shape[2])])

    for i in range(N):
        sm = np.sum([zi[i][k] for k in range(N)])
        for j in range(N):
            zi[i][j] /= sm

   # print(zi)

    
    for i in range(N):
        mean[i] = sum( gamma[i,:] * y)/np.sum(gamma[i,:])
    #print(mean)

    
    y =  np.array(y)
    mean = np.array(mean)
    Z = y.reshape((1,y.shape[0])) - mean.reshape((N,1))
    Z = Z * Z
    var = gamma * Z
    var = np.sum( var , axis =1 ) / np.sum(gamma , axis = 1)


    return zi , mean , var


y = []
temp = ""
mean = []
vari = []
with open("data.txt","r") as t1:
    for L in t1.readlines():
        y.append(float(L))

#print(y)
with open("parameters.txt.txt","r") as t2:
        x = int(t2.readline())                    # x is number of state
        transitions = np.empty((x,x))
        for i in range(x):
            temp = t2.readline().replace("\n" , "")
            temp1 = temp.split("\t")
            for j in range(x):
                transitions[i , j] = temp1[j]
        temp = t2.readline().replace("\n" , "")
        temp1 = temp.split("\t")
        for L in temp1 :
            mean.append(float(L))
        temp = t2.readline().replace("\n" , "")
        temp1 = temp.split("\t")
        for L in temp1 :
            vari.append(float(L))



# emission = []
# for i in range(x):
#     test = []
#     for j in y:
#         prob = pdf(j , mean[i], vari[i])
#         test.append(prob)
#     emission.append(test)

#print(transitions)
#print(emission)
#print(vari)
#print(viterbi(y , transitions , emission))
x = viterbi(y , transitions , mean , vari)
#print(x)
z = []

with open("states_Viterbi_wo_learning.txt","r") as t1:
    for L in t1.readlines():
        L = L.replace("\n" , "")
        L = L.replace("\"" , "")
        L = L.replace("La Nina" , "1")
        L = L.replace("El Nino" , "0")
        z.append(int(L))

#x = google_viterbi(y , transitions, mean , vari)

count = 0
p = []    
for i in range(len(y)):
    if x[i] == z[i]:
        count = count + 1
    else:
        p.append(i)
    
print( "predicted correct : " + str(count) + " accuracy : " + str(count/len(y)))
print(p)

for i in range(20):
    transitions , mean, vari = baum_welch(y , transitions , mean , vari)

print(transitions)
print(mean)
print(vari)