import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from depots import DXY
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#8a8984','axes.linewidth':0.6})
COL={'Depot01':'#2a78d6','Depot02':'#eb6834','Depot03':'#1baf7a','Depot04':'#e87ba4'}
d=pd.read_pickle('deliv.pkl'); d=d[d.dow<5].sample(100000,random_state=1)
fig,ax=plt.subplots(figsize=(6.2,4.6)); x0,y0=d.x.median(),d.y.median()
for dep,c in COL.items():
    g=d[d.depot==dep]; ax.scatter((g.x-x0)/1000,(g.y-y0)/1000,s=0.25,c=c,lw=0,alpha=.5,rasterized=True)
    mx,my=((g.x-x0)/1000).median(),((g.y-y0)/1000).median()
    ax.text(mx,my,dep,fontsize=8.5,weight='bold',color='#0b0b0b',ha='center',va='center',bbox=dict(boxstyle='round,pad=0.25',fc='#fcfcfb',ec=c,lw=1))
for dep,c in COL.items():
    dx,dy=DXY[dep]; ax.scatter((dx-x0)/1000,(dy-y0)/1000,marker='*',s=160,c=c,edgecolor='#0b0b0b',lw=0.8,zorder=5)
ax.scatter([],[],marker='*',s=120,c='#ffffff',edgecolor='#0b0b0b',label='Depot'); ax.legend(loc='upper left',frameon=False,fontsize=8)
qx=((d.x-x0)/1000).quantile([.002,.998]).values; qy=((d.y-y0)/1000).quantile([.002,.998]).values
dxs=[(DXY[k][0]-x0)/1000 for k in COL]; dys=[(DXY[k][1]-y0)/1000 for k in COL]
ax.set_xlim(min(qx[0],min(dxs))-.5,max(qx[1],max(dxs))+.5); ax.set_ylim(min(qy[0],min(dys))-.7,max(qy[1],max(dys))+.5); ax.set_aspect('equal')
ax.set_xlabel('km (east, HTRS96/TM, relative)'); ax.set_ylabel('km (north)')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8,colors='#52514e'); fig.tight_layout()
out=Path(__file__).resolve().parent/'output'; out.mkdir(exist_ok=True); fig.savefig(out/'Figure_1_study_area.png',dpi=300)
