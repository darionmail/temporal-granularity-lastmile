import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm
def hl_ci(x,alpha=0.05):
    x=np.asarray(x); n=len(x); i,j=np.triu_indices(n); W=np.sort((x[i]+x[j])/2)
    N=len(W); mu=n*(n+1)/4; sd=np.sqrt(n*(n+1)*(2*n+1)/24); z=norm.ppf(1-alpha/2)
    k=int(np.floor(mu-z*sd)); return np.median(W),W[max(k,0)],W[min(N-1-k,N-1)]
def paired(a,b):
    x=np.asarray(a)-np.asarray(b); x=x[~np.isnan(x)]
    nz=x[x!=0]; res=wilcoxon(x,zero_method='wilcox')
    rp=np.sum(np.argsort(np.argsort(np.abs(nz)))[nz>0]+1); rn=np.sum(np.argsort(np.argsort(np.abs(nz)))[nz<0]+1)
    rbc=(rp-rn)/(rp+rn) if len(nz) else 0
    hl,lo,hi=hl_ci(x); return dict(n=len(x),median_diff=np.median(x),HL=hl,CI_lo=lo,CI_hi=hi,W=res.statistic,p=res.pvalue,r_rb=rbc)
