import numpy as np
import scipy as sp
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from types import SimpleNamespace


import ipywidgets as widgets

SimpleNamespace()


class ISLM_solve_num:
    def __init__(self):
        









class analytics_IS_LM:
    def Consumption(a,b,Y,T):
        return a+b*(Y-T)
    
    def IS(Y,G=1000,T=10,a=0.8,b=0.5,c=0.2,d=0.8):
        return print(f'r = {(G-T*b+Y*b-Y+a+c)/d}')

    def LM(Y,M=100,P=1,e=1.2,f=0.2):
        return print(f'r = {(-M+P*Y*e)/(P*f)}')

    def AD(G=1000,T=10,M=100,P=1,a=0.8,b=0.5,c=0.2,d=0.8,e=1.2,f=0.2):
        return print(f'{(G*P*f+M*d-P*T*b*f+P*a*f+P*c*f)/(P*(-b*f+d*e+f))}')
    
class numerical_IS_LM:
    def IS(PE,Y,C,I,G=1000,T=10,r,a=0.8,b=0.5,c=0.2,d=0.8):
       PE = C + I + G
       Y = PE
       C = a+b*(Y-T)
       I = c - d*r
       return print(f'r = {r}')asass
    




IS_graph = []
LM_graph = []

print("IS:")
for d in IS_slope:
    print(f'r = {d:1.3f}')
    IS_graph.append(d)

print("LM:")
for d in LM_slope:
    print(f'r = {d:1.3f}')
    LM_graph.append(d)

# create graph
plt.title("IS-LM")
plt.ylabel("$r$")
plt.xlabel("$Y$")
plt.plot(IS_graph, Y_vec, label="IS", color='blue')
plt.plot(LM_graph, Y_vec, label="LM", color='orange')
plt.legend()


#andet forsøg måske bedre
def IS(Y,G=7,T=4,a=0.8,b=0.5,c=0.2,d=0.8):
    r = (G-T*b+Y*b-Y+a+c)/d
    return r

def LM(Y,M=20,P=1,e=1.2,f=0.2):
    r = (-M+P*Y*e)/(P*f)
    return r

def AD(G=7,T=4,M=20,P=1,a=0.8,b=0.5,c=0.2,d=0.8,e=1.2,f=0.2):
    Y = (G*P*f+M*d-P*T*b*f+P*a*f+P*c*f)/(P*(-b*f+d*e+f))
    return Y

print(f'r = {IS(Y=16.22,)}')
print(f'r = {LM(Y=16.22)}')
print(f'Y = {AD()}')



IS_graph = []
LM_graph = []
Y_values = []

for d in range(0,100,1):
    r = IS(Y=d)
    IS_graph.append(r)
    Y_values.append(d)

for d in range(0,100,1):
    r = LM(Y=d)
    LM_graph.append(r)

plt.title("IS-LM")
plt.ylabel("$r$")
plt.xlabel("$Y$")
plt.plot(IS_graph, Y_values, label="IS", color='blue')
plt.plot(LM_graph, Y_values, label="LM", color='orange')
plt.legend()




def IS_analytic(Y,G=200,T=300,a=1000,b=0.75,c=50,d=100):
    r = (G-T*b+Y*b-Y+a+c)/d
    return r

def LM_analytic(Y,M=1000,P=1,e=0.25,f=50):
    r = (-M+P*Y*e)/(P*f)
    return r

def AD_analytic(G=200,T=300,M=1000,P=1,a=1000,b=0.75,c=50,d=100,e=0.25,f=50):
    Y = (G*P*f+M*d-P*T*b*f+P*a*f+P*c*f)/(P*(-b*f+d*e+f))
    return Y


class test():
        # parameter values for slope
    Y_vec = np.array(range(0,8000,100))
    P_vec = np.array(range(1,15,1))
    G_val = 200
    T_val = 300
    M_val = 1000
    P_val = 1
    a_val = 1000
    b_val = 0.75
    c_val = 50
    d_val = 100
    e_val = 0.25
    f_val = 50

    # IS and LM slopes for a vector of Y values
    IS_slope = IS_lambdify(Y_vec,G_val,T_val,a_val,b_val,c_val,d_val)
    LM_slope = LM_lambdify(Y_vec,M_val,P_val,e_val,f_val)
    # AD curve for a vector of P values
    AD_slope = AD_lambdify(G_val,T_val,M_val,P_vec,a_val,b_val,c_val,d_val,e_val,f_val)

    #print("IS:")
    #for d in IS_slope:
    #  print(f'r = {d:1.3f}')
    #print("LM:")
    #for d in LM_slope:
    #   print(f'r = {d:1.3f}')
    #print("AD:")
    #for d in AD_slope:
    #   print(f'Y = {d:1.3f}')

    #create graphs
    IS_graph = []
    LM_graph = []
    AD_graph = []

    # for each value of Y assign an r value to the lists
    for r in IS_slope:
        IS_graph.append(r)

    for r in LM_slope:
        LM_graph.append(r)

    # for each value of P assign an r value to the lists
    for Y in AD_slope:
        AD_graph.append(Y)

    # create graph for IS-LM
    plt.figure()
    plt.title("IS-LM")
    plt.ylabel("$r$")
    plt.xlabel("$Y$")
    plt.plot(Y_vec,IS_graph, label="IS", color='blue')
    plt.plot(Y_vec,LM_graph, label="LM", color='orange')
    plt.legend()

    # create graph for AD
    plt.figure()
    plt.title("AD")
    plt.ylabel("$P$")
    plt.xlabel("$Y$")
    plt.plot(AD_graph, P_vec, label="AD", color='red')
    plt.show()