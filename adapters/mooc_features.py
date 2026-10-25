from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

P=Path(__file__).resolve().parent
CFG=[f'd{d}_{s}' for d in [1,7,14,28] for s in ['N','NS']]
def cfg(k):d,s=k.split('_');return int(d[1:]),s
def protocol():return json.loads((P.parent/'configs/moocradar_folds.json').read_text('utf8'))

def snapshot(h,cut):
    cut=pd.Timestamp(cut);past=h[h.ts<cut];window=past[past.ts>=cut-pd.Timedelta(days=28)]
    stat=past.groupby(['course_id','exercise_id'],sort=True).is_correct.agg(n='size',s='sum').reset_index()
    stat['bid']=stat.groupby('course_id',sort=False).cumcount()
    cats={c:g.sort_values('bid') for c,g in stat.groupby('course_id',sort=False)}
    groups=window.groupby(['user_id','course_id'],sort=True)
    keys=[key for key,n in groups.size().items() if n>=10 and len(cats[key[1]])>=6]
    future=h[(h.ts>=cut)&(h.ts<cut+pd.Timedelta(days=7))]
    targets=future.groupby(['user_id','course_id'],sort=False).head(1).set_index(['user_id','course_id'])
    label_keys=set(targets.index)
    keys=[k for k in keys if k in label_keys and targets.loc[k].exercise_id in set(cats[k[1]].exercise_id)]
    cases=[]
    for key in keys:
        g=groups.get_group(key);cat=cats[key[1]];ids=cat.exercise_id.to_numpy();lookup={v:i for i,v in enumerate(ids)}
        bid=g.exercise_id.map(lookup).to_numpy(dtype=np.int64)
        day=((g.ts-(cut-pd.Timedelta(days=28))).dt.total_seconds().to_numpy()//86400).astype(np.int64)
        N=np.bincount(bid*28+day,minlength=len(ids)*28).reshape(-1,28)
        S=np.bincount(bid*28+day,weights=g.is_correct.to_numpy(),minlength=len(ids)*28).reshape(-1,28).astype(np.int64)
        st=np.c_[np.log1p(cat.n.to_numpy()),(cat.s.to_numpy()+1)/(cat.n.to_numpy()+2)]
        target=lookup[targets.loc[key].exercise_id]
        cases.append({'key':key,'ids':ids,'N':N,'S':S,'static':st,'target':target,'label_target':target,'y':int(targets.loc[key].is_correct)})
    return cases

def reduce_bins(x,old,new):
    assert new%old==0
    return x.reshape(len(x),28//new,new//old).sum(2)
def features(N,S,st,off,target,k,native_days=(1,1)):
    d,style=cfg(k);N=reduce_bins(N,native_days[0],d)
    if style=='NS':S=reduce_bins(S,native_days[1],d)
    z=off+target
    cl=[np.add.reduceat(N,off),N[z]];rk=[N]
    if style=='NS':cl=[np.add.reduceat(N,off),np.add.reduceat(S,off),N[z],S[z]];rk.append(S)
    return np.column_stack(cl+[st[z]]),np.column_stack(rk+[st])
def matrices(cases):
    sizes=np.array([len(c['N']) for c in cases]);off=np.r_[0,np.cumsum(sizes)[:-1]]
    return np.concatenate([c['N'] for c in cases]),np.concatenate([c['S'] for c in cases]),np.concatenate([c['static'] for c in cases]),off,np.array([c['target'] for c in cases])
def rankings(score,off):
    end=np.r_[off[1:],len(score)]
    return np.array([np.lexsort((np.arange(b-a),-score[a:b]))[:5] for a,b in zip(off,end)])
def scores(cases,p,r):
    y=np.array([c['y'] for c in cases]);target=np.array([c['label_target'] for c in cases]);valid=(y>=0)&(target>=0)
    p=np.asarray(p)[valid];y=y[valid];r=np.asarray(r)[valid];target=target[valid]
    hits=np.any(r==target[:,None],axis=1);ndcg=np.max((r==target[:,None])/np.log2(np.arange(5)+2),axis=1)
    return {'n':int(len(y)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'logloss':float(log_loss(y,p,labels=[0,1])) if len(y) else None,'hit5':float(hits.mean()) if len(y) else None,'ndcg5':float(ndcg.mean()) if len(y) else None}
def raw_predictions(cases,models):
    N,S,st,off,target=matrices(cases);out={}
    for k in CFG:
        xc,xr=features(N,S,st,off,target,k)
        p=models['class'][k].predict_proba(xc)[:,1];r=rankings(models['rank'][k].decision_function(xr),off)
        out[k]={'p':p,'r':r,'quality':scores(cases,p,r)}
    mon=np.c_[np.add.reduceat(N.sum(1),off),np.add.reduceat(S.sum(1),off)]
    return out,mon
