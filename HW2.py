import numpy as np
from scipy.stats import binom
from scipy.stats import norm

# call = Max(s-k,0)
# put = Max(k-s,0)

#Choose for Call option or Put option to decide the payoff func and greek func
option_type = input("Enter Put or Call:")

def European (T,N,sigma,r,S0,K,option_type):
    """
    N= number of binomial iteration
    S0=initial price
    K=strike price
    r = risk free interest rate
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1/u
    p = (np.exp(r*t)-d)/(u-d)
    k = np.arange(0,N+1)
    svals = S0 * u**k * d ** (N-k)
    if option_type == 'Call':
        payoffs=np.maximum(svals-K,0)
    elif option_type == 'Put':
        payoffs=np.maximum(K-svals,0)
    probs = binom.pmf(k,n=N, p=p)
    value = (probs @ payoffs) * np.exp(-N*r*t)
    return value

def American_call(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (np.exp(r*t)-d)/(u-d)
    q = 1 - p

    k = np.arange(0,N+1)
    svals = S0*u**(2*k - N)
    payoffs = np.maximum(svals -K, 0)
    discount = np.exp(-r*t)
    def loop(N, discount, K, payoffs):
        if N > 1:
            N=N-1
            payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
            payoffs = payoffs[:-1]
            return loop(N, discount, K, payoffs)
        
        elif N == 1: 
            payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
            payoffs = payoffs[:-1]
            price = np.maximum(payoffs, S0-K)
            return price

    ACall_result = loop(N, discount, K, payoffs)
    return ACall_result

def American_put(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (np.exp(r*t)-d)/(u-d)
    q = 1 - p

    k = np.arange(0,N+1)
    svals = S0 * u**k * d ** (N-k)
    payoffs = np.maximum(K-svals, 0)
    discount = np.exp(-r*t)
    def loop(N, discount, K, payoffs):
      if N > 1:
        N=N-1
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        return loop(N, discount, K, payoffs)
        
      elif N == 1: 
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        price = np.maximum(payoffs, K-S0)
        return price

    APut_result = loop(N, discount, K, payoffs)
    return APut_result


def greeks(S,K,T,r,sigma,option_type):
    d1 = lambda S,K,r,sigma,T : (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = lambda S,K,r,sigma,T : (np.log(S/K) + (r - sigma**2/2)*T) / (sigma*np.sqrt(T))
    if option_type == "Call":
        delta=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))
        gamma=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))/(S*sigma*np.sqrt(T))
        vega=lambda S,K,r,sigma,T: S*norm.cdf(d1(S,K,r,sigma,T))*np.sqrt(T)
        theta=lambda S,K,r,sigma,T:-(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
        rho=lambda S,K,r,sigma,T:K*T*np.exp(-r*T)*norm.cdf(d2(S,K,r,sigma,T))
    elif option_type == "Put":
        delta=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))-1
        gamma=lambda S,K,r,sigma,T:norm.cdf(d1(S,K,r,sigma,T))/(S*sigma*np.sqrt(T))
        vega=lambda S,K,r,sigma,T: S*norm.cdf(d1(S,K,r,sigma,T))*np.sqrt(T)
        theta=lambda S,K,r,sigma,T:-(S*sigma*norm.cdf(d1(S,K,r,sigma,T)))/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))
        rho=lambda S,K,r,sigma,T:-K*T*np.exp(-r*T)*norm.cdf(-d2(S,K,r,sigma,T))
    return delta(S, K, T, r, sigma),gamma(S, K, T, r, sigma),vega(S, K, T, r, sigma),theta(S, K, T, r, sigma),rho(S, K, T, r, sigma)

