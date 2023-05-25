# exercise 1
# a
    # define variables using sympy
    C,alpha,G,nu,L,kappa,tau,w,w_tilde = sm.symbols("C alpha G nu L kappa tau w w_tilde")

    # define w_tilde
    # define the constraint
    C = kappa+(1-tau)*w*L
    # define the maximization problem
    V = sm.log(C**(alpha)*G**(1-alpha))-nu*L**(2)/2
    display(V)
    display(C)

    # find the derirative wrt. L
    dVdL = sm.diff(V,L)
    display(dVdL)
    sol = sm.solve(sm.Eq(dVdL,0), L)
    display(sol[0])

