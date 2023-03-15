
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdOptimizationClass:
    def __init__(self):
        """setting up variables in model
        par: parameters
        sol: solutions"""

        #a. create namespaces
        par=self.par=SimpleNamespace()
        sol=self.sol=SimpleNamespace()

        #b define variables
        #b (i) preferences
        par.rho=2.0
        par.nu=0.001
        par.epsilon=1.0
        par.omega=0.1

        #b (ii) household production
        par.alpha=0.5
        par.sigma=1.0

        #b (iii) wages
        par.w_F=1.0
        par.w_M=1.0

        #c 
##ting at lave om i maxutility måske
    def maxutility(self,L_M,H_M,L_F,H_F):
        """calculates utility for households"""

        par=self.par
        sol=self.sol

        #a consumption of market goods
        C = par.w_M*L_M+par.w_F*L_F

        #b function for hours working at home
        if par.sigma==0:
            H = np.fmin(H_M,H_F)
        elif par.sigma==1:
            H = H_M**(1-par.alpha)*H_F**par.alpha
        else:
            H = (1-par.alpha)*H_M**((par.sigma-1)/par.sigma)+par.alpha*H_F**(par.sigma/(par.sigma-1))

        #c total consumption
        Q = C**par.omega*H**(1-par.omega)

        #d total hours working
        T_M = L_M+H_M
        T_F = L_F+H_F

        #utility function
        #e (i) utility
        utility = Q**(1-par.rho)/(1-par.rho)
            ###måske skal jeg lave dette om hvis jeg får fejl
        #e (ii) disutility
        disutility = par.nu*(T_M**(1+1/par.epsilon)/(1+1/par.epsilon)+T_F**(1+1/par.epsilon)/(1+1/par.epsilon))
        
        return utility-disutility

    def solve_discrete(self,do_print=False):
        """solve model discretely
        opt: optimal values"""
        par=self.par
        sol=self.sol
        opt=SimpleNamespace

        #a possible work combinations
        x = np.linspace(0,24,49)
        L_M,H_M,L_F,H_F = np.meshgrid(x,x,x,x)

        L_M = L_M.ravel()
        H_M = H_M.ravel()
        L_F = L_F.ravel()
        H_F = H_F.ravel()

        #b utility calculation
        u = self.maxutility(L_M,H_M,L_F,H_F)

        #c set constraints
        I = L_M+H_M>24 or L_F+H_F>24
        u[I]=-np.inf #if hours worked for each individual surpasses 24, set utility to -infinity

        #d maximize
        j = np.argmax()

        opt.L_M = L_M[j]
        opt.H_M = H_M[j]
        opt.L_F = L_F[j]
        opt.H_F = H_F[j]

        #print results
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k}={v:6.4f}')
        
        return opt

    def solve_continued(self,do_print=False):
        pass


