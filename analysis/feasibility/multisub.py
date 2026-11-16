# Feasibility check: within-organisation spread of multiple submissions on identical items (tournament).
import sys, pandas as pd, numpy as np
L = pd.read_parquet(sys.argv[1], columns=['file','org','due','id','source','direction','res_date','resolved_to','resolved','forecast','imputed'])
T = L[(L.org!='ForecastBench') & (L.resolved==True) & (L.imputed==False) & (L.direction=='') & L.resolved_to.isin([0.0,1.0])].copy()
T['q']=T.due+'|'+T.source+'|'+T.id+'|'+T.res_date.astype(str); T['bs']=(T.forecast.clip(0,1)-T.resolved_to)**2
out=[]
for (o,d),g in T.groupby(['org','due']):
    fs=g.file.unique()
    if len(fs)<2: continue
    common=set.intersection(*[set(g[g.file==f].q) for f in fs])
    if len(common)<200: continue
    b=g[g.q.isin(common)].groupby('file').bs.mean()
    out.append((o,d,len(fs),len(common),b.max()-b.min(),b.mean()-b.min()))
O=pd.DataFrame(out,columns=['org','due','k','n','range','mean_minus_best'])
print('org-rounds with >=2 submissions on >=200 common resolved items:',len(O),'orgs',O.org.nunique(),'median k',O.k.median())
print('Brier best-vs-mean gap: median',round(O.mean_minus_best.median(),4),'p75',round(O.mean_minus_best.quantile(.75),4),' range median',round(O.range.median(),4))
