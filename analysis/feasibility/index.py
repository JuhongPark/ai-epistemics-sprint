import json, os, sys, collections
import pandas as pd
root = sys.argv[1]; out = sys.argv[2]
rows = []; long = []
for d in sorted(os.listdir(root)):
    dp = os.path.join(root, d)
    if not os.path.isdir(dp): continue
    for fn in sorted(os.listdir(dp)):
        if not fn.endswith('.json'): continue
        with open(os.path.join(dp, fn)) as fh:
            j = json.load(fh)
        fc = j.get('forecasts', [])
        keys = sorted(set(k for f in fc for k in f.keys()))
        src = collections.Counter(f.get('source') for f in fc)
        rows.append(dict(file=fn, dir=d, organization=j.get('organization'), model=j.get('model'),
            model_organization=j.get('model_organization'), due=j.get('forecast_due_date'),
            qset=j.get('question_set'), lb=j.get('leaderboard_eligible'), n=len(fc),
            n_res=sum(1 for f in fc if f.get('resolved')), n_imp=sum(1 for f in fc if f.get('imputed')),
            topkeys='|'.join(sorted(k for k in j.keys() if k!='forecasts')), fkeys='|'.join(keys),
            sources='|'.join(f"{k}:{v}" for k,v in src.most_common())))
        for f in fc:
            dirn = f.get('direction')
            long.append((fn, j.get('model'), j.get('model_organization'), j.get('organization'), j.get('forecast_due_date'),
                str(f.get('id')), f.get('source'), json.dumps(dirn) if dirn else '', f.get('resolution_date'),
                f.get('resolved_to'), f.get('resolved'), f.get('forecast'), f.get('imputed'), f.get('market_value_on_due_date')))
pd.DataFrame(rows).to_csv(os.path.join(out, 'index.csv'), index=False)
L = pd.DataFrame(long, columns=['file','model','morg','org','due','id','source','direction','res_date','resolved_to','resolved','forecast','imputed','mkt'])
for c in ['resolved_to','forecast','mkt']:
    L[c] = pd.to_numeric(L[c], errors='coerce')
L.to_parquet(os.path.join(out, 'long.parquet'), index=False)
print(len(rows), len(L))
