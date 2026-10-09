import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#8a8984','axes.linewidth':0.6})
A=pd.read_csv('rq3f_Depot01_annual.csv'); M=pd.read_csv('rq3f_Depot01.csv'); mons=['Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
for a,(dsg,met,yl) in zip(ax,[('FIX','inband','Fixed plan: districts within ±14% (%)'),('PEN1000','moved_share','Rebalancing: parcels moved from home district (%)')]):
    for src,c,lab,mk in [(M,'#2a78d6','Plan rebuilt monthly','o'),(A,'#eb6834','Plan built in January','s')]:
        s=src[src.design==dsg].groupby('test')[met].mean()*100; a.plot(range(len(s)),s.values,color=c,lw=2,marker=mk,ms=4.5,label=lab)
    a.set_xticks(range(11)); a.set_xticklabels(mons,fontsize=7.5); a.set_ylabel(yl,fontsize=8); a.grid(color='#e6e5e0',lw=.5,axis='y'); a.set_axisbelow(True)
    for s_ in ['top','right']: a.spines[s_].set_visible(False)
    a.tick_params(labelsize=8,colors='#52514e')
ax[0].set_title('(a)',loc='left',fontsize=9); ax[1].set_title('(b)',loc='left',fontsize=9); h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0.08,1,1)); out=Path(__file__).resolve().parent/'output'; out.mkdir(exist_ok=True); fig.savefig(out/'Figure_3_plan_age.png',dpi=300)
