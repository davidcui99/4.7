import pandas as pd
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import stats

rp = pd.read_csv('m_ret_10stocks.txt', sep="\t", header=None, index_col=0)
sp500 = pd.read_csv('m_sp500ret_3mtcm.txt', sep="\t", header=None)
sp500=sp500.dropna(axis='columns')


rm=sp500.iloc[:,1]
rf=sp500.iloc[:,2]
rm=rm.tolist()
rm=sm.add_constant(rm)
rf=rf.tolist()
alpha=[]
beta=[]
residual=[]
sigma0=np.var(rm)
for i in range(10):
    ri=rp.iloc[:,i]
    ri=ri.tolist()
    h=sm.OLS(ri, rm).fit()
    alpha.append(h.params[0])
    beta.append(h.params[1])
    hp=h.resid
    residual.append(h.resid)

residual=pd.DataFrame(residual)
sigma=np.cov(residual.T)
print(sigma)