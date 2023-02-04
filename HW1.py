import numpy as np
from scipy.stats import norm
d1 = lambda S,K,r,sigma,T : (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
d2 = lambda S,K,r,sigma,T : (np.log(S/K) + (r - sigma**2/2)*T) / (sigma*np.sqrt(T))


def BS_call(S, K, r, sigma,T):
    call_price = S * norm.cdf(d1(S,K,r,sigma,T)) - K * np.exp(-r*T)* norm.cdf(d2(S,K,r,sigma,T))
    return call_price
def BS_put(S, K, r, sigma,T):
    put_price = K*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T)) - S*norm.cdf(-d1(S,K,r,sigma,T))
    return put_price
def Call_delta(S,K,r,sigma,T):
    return norm.cdf(d1(S,K,r,sigma,T))
def Put_delta(S,K,r,sigma,T):
    return norm.cdf(d1(S,K,r,sigma,T))-1

def Gamma(S,K,r,sigma,T):
    return norm.cdf(d1(S,K,r,sigma,T))/(S*sigma*np.sqrt(T))

def Vega(S,K,r,sigma,T):
    return S*norm.cdf(d1(S,K,r,sigma,T))*np.sqrt(T)

def Call_Theta(S,K,r,sigma,T):
    return -(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
def Put_Theta(S,K,r,sigma,T):
    return -(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))

def Call_Rho(S,K,r,sigma,T):
    return K*T*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
def Put_Rho(S,K,r,sigma,T):
    return -K*T*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))

