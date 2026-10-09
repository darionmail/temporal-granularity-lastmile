import numpy as np
def nn2opt(P,iters=3):
    n=len(P)
    if n<4: return P
    D=np.sqrt(((P[:,None]-P[None])**2).sum(2)); t=[0]; left=set(range(1,n))
    while left:
        j=min(left,key=lambda k:D[t[-1],k]); t.append(j); left.remove(j)
    t=np.array(t+[0])
    improved=True; it=0
    while improved and it<50:
        improved=False; it+=1
        for i in range(1,n-1):
            a,b=t[i-1],t[i]; c=t[i+1:n]; dd=t[i+2:n+1]
            delta=D[a,c]+D[b,dd]-D[a,b]-D[c,dd]
            j=np.argmin(delta)
            if delta[j]<-1e-6:
                jj=i+1+j; t[i:jj+1]=t[i:jj+1][::-1]; improved=True
    return D[t[:-1],t[1:]].sum()
