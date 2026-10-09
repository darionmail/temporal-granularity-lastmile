import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#8a8984','axes.linewidth':0.6})
R=pd.read_csv('R4road.csv')
S=R.groupby(['depot','design'])[['inband','cont','ridx']].mean().reset_index(); P=R.groupby('design')[['inband','cont','ridx']].mean()
C={'FIX':'#2a78d6','PEN':'#eb6834','DWS':'#1baf7a','DCS':'#eda100','OPR':'#e87ba4'}; MK={'FIX':'s','PEN':'o','DWS':'^','DCS':'v','OPR':'D'}
NAME={'FIX':'Fixed monthly plan','PEN':'Minimal-change rebalancing (λ = 0.25, 1, 5 km)','DWS':'Daily re-optimisation, warm start','DCS':'Daily re-optimisation, cold start','OPR':'Operator (observed)'}
fig,ax=plt.subplots(1,2,figsize=(7.4,3.5))
for a,yv,yl in [(ax[0],'inband','Districts within ±14% of daily mean (%)'),(ax[1],'ridx','Road travel, fixed plan = 100')]:
    for dsg in ['FIX','PEN250','PEN1000','PEN5000','DWS','DCS','OPR']:
        k='PEN' if dsg.startswith('PEN') else dsg; s=S[S.design==dsg]; f=100 if yv=='inband' else 1
        a.scatter(100*s.cont,f*s[yv],s=14,c=C[k],marker=MK[k],alpha=.35,lw=0)
        a.scatter(100*P.loc[dsg,'cont'],f*P.loc[dsg,yv],s=46,c=C[k],marker=MK[k],edgecolor='#fcfcfb',lw=1.2,zorder=3,label=NAME[k] if dsg in('FIX','PEN1000','DWS','DCS','OPR') else None)
    pen=P.loc[['PEN250','PEN1000','PEN5000']]; f=100 if yv=='inband' else 1; a.plot(100*pen.cont,f*pen[yv],color=C['PEN'],lw=1,zorder=2)
    a.set_xlabel('Territory continuity (%)'); a.set_ylabel(yl); a.grid(color='#e6e5e0',lw=.5); a.set_axisbelow(True)
    for s_ in ['top','right']: a.spines[s_].set_visible(False)
    a.tick_params(labelsize=8,colors='#52514e')
ax[1].set_ylim(96,146); ax[0].set_title('(a) Workload balance',loc='left',fontsize=9); ax[1].set_title('(b) Road travel incl. depot drives',loc='left',fontsize=9)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.5,0.04),ncol=3,frameon=False,fontsize=7.5)
fig.text(0.5,0.005,'Large markers: mean of four depots, 916 test weekdays. Small markers: single depots.',ha='center',fontsize=7,color='#52514e')
fig.tight_layout(rect=(0,0.16,1,1)); out=Path(__file__).resolve().parent/'output'; out.mkdir(exist_ok=True); fig.savefig(out/'Figure_2_tradeoff.png',dpi=300)
