import numpy as np
from scipy.spatial import ConvexHull
def two_opt(order,D):
    # order: list of node indices, closed tour starting/ending at order[0]
    t=np.array(order+[order[0]]); n=len(t)-1
    if n<4: return t[:-1]
    improved=True; it=0
    while improved and it<100:
        improved=False; it+=1
        for i in range(1,n-1):
            a,b=t[i-1],t[i]; c=t[i+1:n]; dd=t[i+2:n+1]
            delta=D[a,c]+D[b,dd]-D[a,b]-D[c,dd]
            j=np.argmin(delta)
            if delta[j]<-1e-9:
                jj=i+1+j; t[i:jj+1]=t[i:jj+1][::-1]; improved=True
    return t[:-1]
def or_opt(t,D):
    t=list(t); n=len(t); improved=True; it=0
    while improved and it<50:
        improved=False; it+=1
        for seg in (1,2,3):
            for i in range(1,n-seg+1):
                if i+seg>n: break
                s=t[i:i+seg]; prev=t[i-1]; nxt=t[(i+seg)%n]
                rem=D[prev,s[0]]+D[s[-1],nxt]-D[prev,nxt]
                rest=t[:i]+t[i+seg:]
                best=None;bg=1e-9
                for j in range(len(rest)):
                    a=rest[j]; b=rest[(j+1)%len(rest)]
                    for ss in (s,s[::-1]):
                        add=D[a,ss[0]]+D[ss[-1],b]-D[a,b]
                        if rem-add>bg: bg=rem-add; best=(j,ss)
                if best:
                    j,ss=best; t=rest[:j+1]+list(ss)+rest[j+1:]
                    k=t.index(0); t=t[k:]+t[:k]; improved=True; break
            if improved: break
    return t
def route(nodes,D,T,polish=True):
    """nodes: matrix indices, nodes[0]=depot. Returns (km, minutes, order) using symmetric
    optimisation then evaluating both directions on the asymmetric matrices."""
    m=len(nodes)
    if m==1: return 0.0,0.0,[nodes[0]]
    Ds=(D[np.ix_(nodes,nodes)]+D[np.ix_(nodes,nodes)].T)/2
    # nearest neighbour from depot
    left=set(range(1,m)); o=[0]
    while left:
        j=min(left,key=lambda k:Ds[o[-1],k]); o.append(j); left.remove(j)
    o=list(two_opt(o,Ds))
    if polish and m<=80: o=or_opt(o,Ds)
    k=o.index(0); o=o[k:]+o[:k]
    best=None
    Da=D[np.ix_(nodes,nodes)]; Ta=T[np.ix_(nodes,nodes)]
    for oo in (o,[o[0]]+o[1:][::-1]):
        seq=oo+[oo[0]]; km=sum(Da[a,b] for a,b in zip(seq[:-1],seq[1:])); mn=sum(Ta[a,b] for a,b in zip(seq[:-1],seq[1:]))
        if best is None or km<best[0]: best=(km,mn,[nodes[i] for i in oo])
    return best
def intra(P):
    n=len(P)
    if n<3: return 0.0 if n<2 else float(np.hypot(*(P[0]-P[1])))*2/1000
    try: A=ConvexHull(P).volume
    except Exception: A=0.0
    return 0.7124*np.sqrt(n*A)/1000

def route_multi(nodes,D,T,starts=6,seed=0):
    """Best of several constructions (nearest neighbour from different start nodes), each 2-opt + or-opt."""
    m=len(nodes)
    if m<=3: return route(nodes,D,T)
    Ds=(D[np.ix_(nodes,nodes)]+D[np.ix_(nodes,nodes)].T)/2
    rng=np.random.default_rng(seed); cand=[0]+list(rng.choice(np.arange(1,m),min(starts-1,m-1),replace=False))
    best=None
    for s0 in cand:
        left=set(range(m))-{s0}; o=[s0]
        while left:
            j=min(left,key=lambda k:Ds[o[-1],k]); o.append(j); left.remove(j)
        k=o.index(0); o=o[k:]+o[:k]
        o=list(two_opt(o,Ds)); o=or_opt(o,Ds); o=list(two_opt(o,Ds))
        k=o.index(0); o=o[k:]+o[:k]
        c=sum(Ds[a,b] for a,b in zip(o,o[1:]+o[:1]))
        if best is None or c<best[0]: best=(c,o)
    o=best[1]; Da=D[np.ix_(nodes,nodes)]; Ta=T[np.ix_(nodes,nodes)]; res=None
    for oo in (o,[o[0]]+o[1:][::-1]):
        seq=oo+[oo[0]]; km=sum(Da[a,b] for a,b in zip(seq[:-1],seq[1:])); mn=sum(Ta[a,b] for a,b in zip(seq[:-1],seq[1:]))
        if res is None or km<res[0]: res=(km,mn,[nodes[i] for i in oo])
    return res
