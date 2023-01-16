import pandas as pd
import numpy as np
import math
import scipy.stats as stats


# importing data
fgdata = pd.read_csv(r'~/ST503/Code/nfl2008_fga.csv')

def expit(x):
    val = math.exp(x)/(1+math.exp(x))
    return val;

# create the prior density function
def logprior(beta0,beta1):
    priorbeta0 = stats.norm(0, 1).pdf(beta0)
    priorbeta1 = stats.norm(0, 1).pdf(beta1)
    jointlogdens = math.log(priorbeta0) + math.log(priorbeta1)
    return jointlogdens

def loglikelihood(beta0,beta1,df,x,y):
    loglik = 0
    for ind in df.index:
        p = expit(beta0 + beta1 * (df[x][ind]))
        dens = berndens(df[y][ind],prob = p) # y is success (1) or failure (0) from data, p is probability from line before.
        loglik = loglik + math.log(dens)
    return loglik

# returns bernoulli density for a particular value of x (0 or 1) of a given probability. equiv. to dbinom(x, size=1, prob)
def berndens(x, prob):
    if x == 0:
        return (1-prob)
    else:
        return prob

y = expit(6 + (-0.12 * 45))
z = berndens(1, prob = y)
# y = loglikelihood(beta0 = 6, beta1 = -0.12, lendata=1, x = 45, y = 1)


data = {'GOOD': [1,0,1,1,1],
        'distance':[30,55,28,41,38]}

pretend = pd.DataFrame(data, columns = ['GOOD','distance'])

# z = loglikelihood(beta0 = 6.7, beta1 = -0.12, df = fgdata, x = 'distance', y = 'GOOD')


def betasampler(niter, betaproposalsd, data, betainitialproposal):
    beta0store = np.zeros(niter)
    beta1store = np.zeros(niter) # alternative way of doing this, the 'zero' is important in the math so also ambiguous
    # probably the most generalizable way to do this is would be to pass betainitalproposal in as function parameter
    # then you could say betastore = np.empty(niter) and in the following line betastore[0] = betainitialproposal

    #loop through the desired number of iterations
    for i in range(0,(niter-1)):

        currentbeta0 = beta0store[i]
        currentbeta1 = beta1store[i]

        newbeta0 = np.random.normal(loc = currentbeta0,scale = betaproposalsd)
        newbeta1 = np.random.normal(loc = currentbeta1,scale = betaproposalsd)

        # find the log of r ratio for beta 0
        logaccept = logprior(beta0=newbeta0, beta1=currentbeta1) + loglikelihood(beta0 = newbeta0,beta1=currentbeta1,df = data,x = 'distance',y = 'GOOD') - logprior(beta0=currentbeta0, beta1=currentbeta1) - loglikelihood(beta0 = currentbeta0,beta1=currentbeta1,df = data,x = 'distance',y = 'GOOD')
        if np.random.uniform() < math.exp(logaccept):
            beta0store[i+1] = newbeta0
        else:
            beta0store[i+1] = currentbeta0
            
          # find the log of r ratio for beta 0
        logaccept= logprior(beta0=newbeta0, beta1=newbeta1) + loglikelihood(beta0 = newbeta0,beta1=newbeta1,df = data,x = 'distance',y = 'GOOD') - logprior(beta0=newbeta0, beta1=currentbeta1) - loglikelihood(beta0 = newbeta0,beta1=currentbeta1,df = data,x = 'distance',y = 'GOOD')
        if np.random.uniform() < math.exp(logaccept):
            beta1store[i+1] = newbeta1
        else:
            beta1store[i+1] = currentbeta1    
    data = {'Beta0':[beta0store], 'Beta1':[beta1store]}
    return pd.DataFrame(data)


# niter here will give you a number of draws from the posterior (~5k is usually enough for inference)
df = betasampler(niter=200, betaproposalsd = 0.1, betainitialproposal = 0, data=pretend)

print (df.mean(axis=0))

