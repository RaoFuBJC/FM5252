import numpy as np
from scipy.stats import norm
d1 = lambda S,K,r,sigma,T : (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
d2 = lambda S,K,r,sigma,T : (np.log(S/K) + (r - sigma**2/2)*T) / (sigma*np.sqrt(T))


call_price=lambda S,K,r,sigma,T: S * norm.cdf(d1(S,K,r,sigma,T)) - K * np.exp(-r*T)* norm.cdf(d2(S,K,r,sigma,T))

put_price =lambda S,K,r,sigma,T: K*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T)) - S*norm.cdf(-d1(S,K,r,sigma,T))

Call_delta=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))
Put_delta=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))-1

Gamma=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))/(S*sigma*np.sqrt(T))

Vega=lambda S,K,r,sigma,T: S*norm.cdf(d1(S,K,r,sigma,T))*np.sqrt(T)

Call_Theta=lambda S,K,r,sigma,T:-(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
Put_Theta=lambda S,K,r,sigma,T:-(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))

Call_Rho=lambda S,K,r,sigma,T:K*T*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
Put_Rho=lambda S,K,r,sigma,T:-K*T*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))

