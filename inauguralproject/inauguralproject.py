
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

# Original model
class HouseholdOptimizationClass:
    def __init__(self):
        """setting up variables in model
        par: parameters
        sol: solutions"""

        # a. create namespaces
        par=self.par=SimpleNamespace()
        sol=self.sol=SimpleNamespace()

        # b define variables
        #b (i) preferences
        par.rho=2.0
        par.nu=0.001
        par.epsilon=1.0
        par.omega=0.5

        #b (ii) household production
        par.alpha=0.5
        par.sigma=1.0

        #b (iii) wages
        par.w_F=1.0
        par.w_M=1.0
        par.w_F_vec=np.linspace(0.8,1.2,5)

        #c beta targets
        par.beta0_target=0.4
        par.beta1_target=-0.1

        #e solution
        sol.L_M_vec = np.zeros(par.w_F_vec.size)
        sol.H_M_vec = np.zeros(par.w_F_vec.size)
        sol.L_F_vec = np.zeros(par.w_F_vec.size)
        sol.H_F_vec = np.zeros(par.w_F_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

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
            H = ((1-par.alpha)*H_M**((par.sigma-1)/par.sigma)+par.alpha*H_F**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        #c total consumption
        Q = C**par.omega*H**(1-par.omega)

        #d total hours working
        T_M = L_M+H_M
        T_F = L_F+H_F

        # utility function
        #e (i) utility
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
        #e (ii) disutility
        disutility = par.nu*(T_M**(1+1/par.epsilon)/(1+1/par.epsilon)+T_F**(1+1/par.epsilon)/(1+1/par.epsilon))
        
        return utility-disutility

    def solve_discrete(self,do_print=False):
        """solve model discretely
        opt: optimal values"""
        par=self.par
        sol=self.sol
        opt=SimpleNamespace()

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
        I = (L_M+H_M > 24) | (L_F+H_F > 24)
        u[I]=-np.inf #if hours worked for each individual surpasses 24, set utility to -infinity

        #d maximize
        j = np.argmax(u)

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
        """solves the model in continued time using optimize function"""
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        #create objective function
        def objective(x): 
            L_M, H_M, L_F, H_F = x
            return -self.maxutility(L_M, H_M, L_F, H_F)*100 # scale the function by 100 to get more accurate results
        
        obj = lambda x: objective(x)
        # set constraints
        constraint_m= lambda x: 24-x[0]-x[1] #ensures total work for each person does not go above 24
        constraint_f= lambda x: 24-x[2]-x[3]
        constraints = ({'type':'ineq','fun': constraint_m},{'type':'ineq','fun':constraint_f})
        # initial guess for L and H variables
        guess = [4.5]*4
        # bounds for L and H variables
        bounds = [(0,24)]*4

        # optimization
        result = optimize.minimize(obj,guess,method='SLSQP',bounds=bounds,constraints=constraints)

        # save output
        L_M_opt, H_M_opt, L_F_opt, H_F_opt = result.x

        opt.L_M = L_M_opt
        opt.H_M = H_M_opt
        opt.L_F = L_F_opt
        opt.H_F = H_F_opt
        

        #prints results if requested
        if do_print:
            print("Optimal Solution:")
            print("L_M = {:.2f}".format(opt.L_M))
            print("H_M = {:.2f}".format(opt.H_M))
            print("L_F = {:.2f}".format(opt.L_F))
            print("H_F = {:.2f}".format(opt.H_F))
    

        return opt

    def solve_w_F_vec(self,discrete=False):
        """saves the optimal values of H and L for w_F(0.8,1.2,5)"""
        par=self.par
        sol=self.sol
        
        for i,w_F in enumerate(par.w_F_vec):
            #assigns each value of the vector to w_F
            par.w_F=w_F
            
            #solves and stores values in vector for continued time
            opt=self.solve_continued()

            sol.H_F_vec[i]=opt.H_F
            sol.H_M_vec[i]=opt.H_M
            sol.L_F_vec[i]=opt.L_F
            sol.L_M_vec[i]=opt.L_M

        return sol.H_F_vec,sol.H_M_vec,sol.L_F_vec,sol.L_M_vec
        
    def regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.w_F_vec)
        y = np.log(sol.H_F_vec/sol.H_M_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def estimate(self,alpha=None,sigma=None):
        """estimate the alpha and sigma values to minimize beta differences"""
        par=self.par
        sol=self.sol
        opt=self.opt=SimpleNamespace()
        #create the function to minimize
        def min_diff(x):
            par.alpha, par.sigma=x
            self.solve_w_F_vec()
            self.regression()
            return (par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2
    
        # use the initial alpha and sigma values as guesses
        guess = (par.alpha,par.sigma)
        # values alpha and sigma can take
        bounds = [(0,1),(0,1.5)]
        result = optimize.minimize(min_diff,guess,method="Nelder-Mead",bounds=bounds)
        
        opt.alpha, opt.sigma = result.x
        # print solutions
        print(f' beta differences: {(par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2:.3}')
        print(f' alpha: {opt.alpha:.3}, sigma: {opt.sigma:.3}, beta0: {sol.beta0:.3}, beta1: {sol.beta1:.3}')

        return opt.alpha, opt.sigma

#The model with extra disutility for home production for men in the utility function
class Modelchange:
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
        par.omega=0.5

        #b (ii) household production
        par.alpha=0.5
        par.sigma=1.0

        #b (iii) wages
        par.w_F=1.0
        par.w_M=1.0
        par.w_F_vec=np.linspace(0.8,1.2,5)

        #c beta targets
        par.beta0_target=0.4
        par.beta1_target=-0.1

        #e solution
        sol.L_M_vec = np.zeros(par.w_F_vec.size)
        sol.H_M_vec = np.zeros(par.w_F_vec.size)
        sol.L_F_vec = np.zeros(par.w_F_vec.size)
        sol.H_F_vec = np.zeros(par.w_F_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

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
            H = ((1-par.alpha)*H_M**((par.sigma-1)/par.sigma)+par.alpha*H_F**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        #c total consumption
        Q = C**par.omega*H**(1-par.omega)

        #d total hours working
        T_M = L_M+H_M
        T_F = L_F+H_F

        #utility function
        #e (i) utility
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
        #e (ii) disutility
        disutility = par.nu*(T_M**(1+1/par.epsilon)/(1+1/par.epsilon)+T_F**(1+1/par.epsilon)/(1+1/par.epsilon))
        
        #disutility for men
        men_disutility=55*par.nu*H_M**(1+1/par.epsilon)/(1+1/par.epsilon)

        return utility-disutility-men_disutility

    def solve_discrete(self,do_print=False):
        """solve model discretely
        opt: optimal values"""
        par=self.par
        sol=self.sol
        opt=SimpleNamespace()

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
        I = (L_M+H_M > 24) | (L_F+H_F > 24)
        u[I]=-np.inf #if hours worked for each individual surpasses 24, set utility to -infinity

        #d maximize
        j = np.argmax(u)

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
        """solves the model in continued time using optimize function"""
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        #create objective function
        def objective(x): 
            L_M, H_M, L_F, H_F = x
            return -self.maxutility(L_M, H_M, L_F, H_F)*100 #we scale the function by 100 to get more accurate results
        
        obj = lambda x: objective(x)
        #set constraints
        constraint_m= lambda x: 24-x[0]-x[1] #ensures total work for each person does not go above 24
        constraint_f= lambda x: 24-x[2]-x[3]
        constraints = ({'type':'ineq','fun': constraint_m},{'type':'ineq','fun':constraint_f})
        #initial guess for L and H variables
        guess = [4.5]*4
        #bounds for L and H variables
        bounds = [(0,24)]*4

        #optimization
        result = optimize.minimize(obj,guess,method='SLSQP',bounds=bounds,constraints=constraints)

        #save output
        L_M_opt, H_M_opt, L_F_opt, H_F_opt = result.x

        opt.L_M = L_M_opt
        opt.H_M = H_M_opt
        opt.L_F = L_F_opt
        opt.H_F = H_F_opt
        

        #prints results if requested
        if do_print:
            print("Optimal Solution:")
            print("L_M = {:.2f}".format(opt.L_M))
            print("H_M = {:.2f}".format(opt.H_M))
            print("L_F = {:.2f}".format(opt.L_F))
            print("H_F = {:.2f}".format(opt.H_F))
    

        return opt

    def solve_w_F_vec(self,discrete=False):
        """saves the optimal values of H and L for w_F(0.8,1.2,5)"""
        par=self.par
        sol=self.sol
        
        for i,w_F in enumerate(par.w_F_vec):
            #assigns each value of the vector to w_F
            par.w_F=w_F
            
            #solves and stores values in vector for continued time
            opt=self.solve_continued()

            sol.H_F_vec[i]=opt.H_F
            sol.H_M_vec[i]=opt.H_M
            sol.L_F_vec[i]=opt.L_F
            sol.L_M_vec[i]=opt.L_M

        return sol.H_F_vec,sol.H_M_vec,sol.L_F_vec,sol.L_M_vec
        
    def regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.w_F_vec)
        y = np.log(sol.H_F_vec/sol.H_M_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0,sol.beta1

    def estimate(self,alpha=None,sigma=None):
        """estimate the alpha and sigma values to minimize beta differences"""
        par=self.par
        sol=self.sol
        opt = self.opt = SimpleNamespace()
        #create the function to minimize
        def min_diff(x):
            par.sigma=x
            self.solve_w_F_vec()
            self.regression()
            return (par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2
    
        #use the initial sigma value as guess
        guess = (par.sigma)
        #values sigma can take
        bounds = [(0,1.5)]
        result = optimize.minimize(min_diff,guess,method="Nelder-Mead",bounds=bounds)
        
        opt.sigma = result.x
        print(f' beta differences: {(par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2:.3}')
        print(f' alpha: {par.alpha:.3}, sigma: {opt.sigma}, beta0: {sol.beta0:.3}, beta1: {sol.beta1:.3}')
        
        return opt.sigma

#Original model but with a fixed alpha of 0.5
class OGModelFixedAlpha:
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
        par.omega=0.5

        #b (ii) household production
        par.alpha=0.5
        par.sigma=1.0

        #b (iii) wages
        par.w_F=1.0
        par.w_M=1.0
        par.w_F_vec=np.linspace(0.8,1.2,5)

        #c beta targets
        par.beta0_target=0.4
        par.beta1_target=-0.1

        #e solution
        sol.L_M_vec = np.zeros(par.w_F_vec.size)
        sol.H_M_vec = np.zeros(par.w_F_vec.size)
        sol.L_F_vec = np.zeros(par.w_F_vec.size)
        sol.H_F_vec = np.zeros(par.w_F_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

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
            H = ((1-par.alpha)*H_M**((par.sigma-1)/par.sigma)+par.alpha*H_F**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        #c total consumption
        Q = C**par.omega*H**(1-par.omega)

        #d total hours working
        T_M = L_M+H_M
        T_F = L_F+H_F

        #utility function
        #e (i) utility
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)
        #e (ii) disutility
        disutility = par.nu*(T_M**(1+1/par.epsilon)/(1+1/par.epsilon)+T_F**(1+1/par.epsilon)/(1+1/par.epsilon))
        
        return utility-disutility

    def solve_discrete(self,do_print=False):
        """solve model discretely
        opt: optimal values"""
        par=self.par
        sol=self.sol
        opt=SimpleNamespace()

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
        I = (L_M+H_M > 24) | (L_F+H_F > 24)
        u[I]=-np.inf #if hours worked for each individual surpasses 24, set utility to -infinity

        #d maximize
        j = np.argmax(u)

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
        """solves the model in continued time using optimize function"""
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        #create objective function
        def objective(x): 
            L_M, H_M, L_F, H_F = x
            return -self.maxutility(L_M, H_M, L_F, H_F)*100 #we scale the function by 100 to get more accurate results
        
        obj = lambda x: objective(x)
        #set constraints
        constraint_m= lambda x: 24-x[0]-x[1] #ensures total work for each person does not go above 24
        constraint_f= lambda x: 24-x[2]-x[3]
        constraints = ({'type':'ineq','fun': constraint_m},{'type':'ineq','fun':constraint_f})
        #initial guess for L and H variables
        guess = [4.5]*4
        #bounds for L and H variables
        bounds = [(0,24)]*4

        #optimization
        result = optimize.minimize(obj,guess,method='SLSQP',bounds=bounds,constraints=constraints)

        #save output
        L_M_opt, H_M_opt, L_F_opt, H_F_opt = result.x

        opt.L_M = L_M_opt
        opt.H_M = H_M_opt
        opt.L_F = L_F_opt
        opt.H_F = H_F_opt
        

        #prints results if requested
        if do_print:
            print("Optimal Solution:")
            print("L_M = {:.2f}".format(opt.L_M))
            print("H_M = {:.2f}".format(opt.H_M))
            print("L_F = {:.2f}".format(opt.L_F))
            print("H_F = {:.2f}".format(opt.H_F))
    

        return opt

    def solve_w_F_vec(self,discrete=False):
        """saves the optimal values of H and L for w_F(0.8,1.2,5)"""
        par=self.par
        sol=self.sol
        
        for i,w_F in enumerate(par.w_F_vec):
            #assigns each value of the vector to w_F
            par.w_F=w_F
            
            #solves and stores values in vector for continued time
            opt=self.solve_continued()

            sol.H_F_vec[i]=opt.H_F
            sol.H_M_vec[i]=opt.H_M
            sol.L_F_vec[i]=opt.L_F
            sol.L_M_vec[i]=opt.L_M

        return sol.H_F_vec,sol.H_M_vec,sol.L_F_vec,sol.L_M_vec
        
    def regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.w_F_vec)
        y = np.log(sol.H_F_vec/sol.H_M_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0,sol.beta1

    def estimate(self,alpha=None,sigma=None):
        """estimate the alpha and sigma values to minimize beta differences"""
        par=self.par
        sol=self.sol
        opt = self.opt = SimpleNamespace()
        #create the function to minimize
        def min_diff(x):
            par.sigma=x
            self.solve_w_F_vec()
            self.regression()
            return (par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2
    
        #use the initial sigma value as guess
        guess = (par.sigma)
        #values sigma can take
        bounds = [(0,1.5)]
        result = optimize.minimize(min_diff,guess,method="Nelder-Mead",bounds=bounds)
        
        opt.sigma = result.x
        print(f' beta differences: {(par.beta0_target-sol.beta0)**2+(par.beta1_target-sol.beta1)**2:.3}')
        print(f' alpha: {par.alpha:.3}, sigma: {opt.sigma}, beta0: {sol.beta0:.3}, beta1: {sol.beta1:.3}')

        return opt.sigma