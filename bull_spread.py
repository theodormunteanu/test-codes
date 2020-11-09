# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:39:56 2020

@author: Theodor
"""

def bull_spread_payoff_test():
    r"""
    Payoff of a bull call spread with strike prices $K_1=90,K_2 = 100$
    """
    def payoff(x):
        if x<90:
            return 0
        elif 90<=x and x<=100: 
            return x-90
        else:
            return 10
    import matplotlib.pyplot as plt
    div = list(range(40,141))
    T = 6
    plt.figure()
    plt.plot(div,[payoff(x) for x in div])
    plt.xlabel('Spot price in {0} months'.format(T))
    plt.ylabel('Payoff')
    plt.title('Payoff of a bull spread, $K_1 = 90, K_2 =100$')
    plt.grid(True)
    plt.show()
bull_spread_payoff_test()
#%%
def test_exercise2():
    import numpy as np
    dt,sig,S0 = 1/12,0.1,4.8
    u= np.exp(sig*np.sqrt(dt))
    d= 1/u
    p = (np.exp(0.01*dt)-d)/(u-d)
    print(u,d,p)
    S2 = [S0*u**2,S0,S0*d**2]
    probs = [p**2,2*p*(1-p),(1-p)**2]
    payoffs = [max(x-4.85,0) for x in S2]
    price = np.exp(-0.02*2/12)* np.dot(payoffs,probs)
    print("Probabilities",probs)
    print("Payoffs",payoffs)
    print(price*2000)
test_exercise2()

#%%
def test_BS_exercise1c():
    S0,r,K1,K2,T,sig1,sig2,sig3 = 100,0,90,100,0.5,0.1,0.2,0.3
    import numpy as np
    import scipy.stats as stats
    def d12(K,sig):
        d1 = (np.log(S0/K)+(r+sig**2/2)*T)/(sig*np.sqrt(T))
        d2 = d1-sig*np.sqrt(T)
        return d1,d2
    price1 = S0*(stats.norm.cdf(d12(K1,sig1)[0])-stats.norm.cdf(d12(K2,sig1)[0]))-\
             np.exp(-r*T)*(K1*stats.norm.cdf(d12(K1,sig1)[1])-K2*stats.norm.cdf(d12(K2,sig1)[1]))
    price2 = S0*(stats.norm.cdf(d12(K1,sig2)[0])-stats.norm.cdf(d12(K2,sig2)[0]))-\
             np.exp(-r*T)*(K1*stats.norm.cdf(d12(K1,sig2)[1])-K2*stats.norm.cdf(d12(K2,sig2)[1]))
    price3 = S0*(stats.norm.cdf(d12(K1,sig3)[0])-stats.norm.cdf(d12(K2,sig3)[0]))-\
             np.exp(-r*T)*(K1*stats.norm.cdf(d12(K1,sig3)[1])-K2*stats.norm.cdf(d12(K2,sig3)[1]))
    print(price1)
    print(price2)
    print(price3)
test_BS_exercise1c()
#%%
def test_binomial_exercise1d():
    S0,r,K1,K2,T,sig1,sig2,sig3,dt  = 100,0,90,100,0.5,0.1,0.2,0.3,0.25
    import numpy as np 
    u1,u2,u3 = np.exp(sig1*np.sqrt(dt)),np.exp(sig2*np.sqrt(dt)),np.exp(sig3*np.sqrt(dt))
    d1,d2,d3 = 1/u1,1/u2,1/u3
    p1,p2,p3 = [(1-d1)/(u1-d1),(1-d2)/(u2-d2),(1-d3)/(u3-d3)]
    S21,S22,S23 = [S0*u1**2,S0,S0*d1**2],[S0*u2**2,S0,S0*d2**2],[S0*u3**2,S0,S0*d3**2]
    payoffs1,payoffs2,payoffs3 = [max([x-90,0])-max([x-100,0]) for x in S21],\
                                 [max([x-90,0])-max([x-100,0]) for x in S22],\
                                 [max([x-90,0])-max([x-100,0]) for x in S23]
    probs1,probs2,probs3 = [p1**2,2*p1*(1-p1),(1-p1)**2],[p2**2,2*p2*(1-p2),(1-p2)**2],\
                           [p3**2,2*p3*(1-p3),(1-p3)**2]
    price1,price2,price3 = np.dot(payoffs1,probs1),np.dot(payoffs2,probs2),np.dot(payoffs3,probs3)
    print(price1,price2,price3)
test_binomial_exercise1d()