import pandas as pd
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import stats
from sklearn.metrics import mean_squared_error

rp = pd.read_csv('m_ret_10stocks.txt', sep="\t", header=None, index_col=0)
sp500 = pd.read_csv('m_sp500ret_3mtcm.txt', sep="\t", header=None)
sp500=sp500.dropna(axis='columns')

# a)

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
# F=sigma0*beta*beta.T+sigma*identity matrix


# b)
s=np.cov(rp.T)
pi_hatij=[]
r_matrix=[]
h_array=[]
h_arr=[]
for i in range(10):
    r_matrix.append((rp.iloc[:,i]-rp.mean(axis=1)).tolist())

r_matrix=pd.DataFrame(r_matrix)




# for j in range(156):
#     b = matrix(r_matrix.iloc[:, j],tc='d')
# Do the cross product with itself. should be 156 matrices of size 10*10. Put them in a list and then do mse with respect to s

