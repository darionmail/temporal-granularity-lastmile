import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
LOG=[]
def kpp(X,w,K,rng):
    p=w/w.sum(); C=[X[rng.choice(len(X),p=p)]]
    for _ in range(K-1):
        D=np.min(((X[:,None]-np.array(C)[None])**2).sum(2),1)*w
        C.append(X[rng.choice(len(X),p=D/D.sum())])
    return np.array(C)
def assign_lp(X,w,C,eps=0.05,home=None,lam=0.0,integer=False):
    n,K=len(X),len(C); W=w.sum(); lo,hi=W/K*(1-eps),W/K*(1+eps)
    Dm=np.sqrt(((X[:,None]-C[None])**2).sum(2))
    if home is not None: Dm=Dm+lam*(np.arange(K)[None,:]!=home[:,None])
    cost=(Dm*w[:,None]).ravel()
    rows=np.repeat(np.arange(n),K); cols=np.arange(n*K)
    Aeq=sp.csr_matrix((np.ones(n*K),(rows,cols)),shape=(n,n*K))
    kk=np.tile(np.arange(K),n)
    Aub=sp.vstack([sp.csr_matrix((w[rows],(kk,cols)),shape=(K,n*K)),sp.csr_matrix((-w[rows],(kk,cols)),shape=(K,n*K))])
    bub=np.r_[np.full(K,hi),np.full(K,-lo)]
    if integer:
        from scipy.optimize import milp, LinearConstraint, Bounds
        r=milp(cost,constraints=[LinearConstraint(Aeq,np.ones(n),np.ones(n)),LinearConstraint(Aub[:K],-np.inf,np.full(K,hi)),LinearConstraint(-Aub[K:],np.full(K,lo),np.inf)],
               integrality=np.ones(n*K),bounds=Bounds(0,1),options=dict(time_limit=10,mip_rel_gap=1e-3))
        ok=r.x is not None
        LOG.append(dict(status=r.status,message=str(r.message)[:60],gap=getattr(r,'mip_gap',np.nan),fallback=not ok))
        if ok: return r.x.reshape(n,K).argmax(1)
    r=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=np.ones(n),bounds=(0,None),method='highs')
    if r.status!=0:
        return np.argmin(((X[:,None]-C[None])**2).sum(2),1)
    return r.x.reshape(n,K).argmax(1)
def cap_kmeans(X,w,K,C0=None,rng=None,iters=6,eps=0.05):
    C=kpp(X,w,K,rng) if C0 is None else C0.copy()
    for _ in range(iters):
        lab=assign_lp(X,w,C,eps)
        Cn=np.array([np.average(X[lab==k],axis=0,weights=w[lab==k]) if (lab==k).any() else C[k] for k in range(K)])
        if np.abs(Cn-C).max()<1: C=Cn; break
        C=Cn
    return assign_lp(X,w,C,eps),C
