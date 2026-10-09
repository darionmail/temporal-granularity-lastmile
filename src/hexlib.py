import numpy as np
S3=np.sqrt(3)
N=np.array([[1,0],[0.5,S3/2],[-0.5,S3/2]])  # pointy-top edge normals
def snap(x,y,L=1000.0):
    q=(S3/3*x - y/3)/L; r=(2/3*y)/L
    cx,cz=q,r; cy=-cx-cz
    rx,ry,rz=np.round(cx),np.round(cy),np.round(cz)
    dx,dy,dz=abs(rx-cx),abs(ry-cy),abs(rz-cz)
    if dx>dy and dx>dz: rx=-ry-rz
    elif dy>dz: ry=-rx-rz
    else: rz=-rx-ry
    return L*S3*(rx+rz/2), L*1.5*rz
def hexnorm(P,c):
    # side length needed for exact containment of each point
    D=P-c; return np.abs(D@N.T).max(1)/(S3/2)
def snap_cover_cap(P,center=None,L=1000.0,smin=750.0,smax=2000.0,exact=False):
    c=P.mean(0) if center is None else center
    c=np.array(snap(c[0],c[1],L))
    need = hexnorm(P,c).max() if exact else (2/S3)*np.sqrt(((P-c)**2).sum(1)).max()
    s=min(max(need,smin),smax); return c,s
def inside(P,c,s): return hexnorm(P,c)<=s+1e-9
