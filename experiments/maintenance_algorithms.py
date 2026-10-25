"""Batch, bounded, selective, and adaptive correction policies.

Each policy reads the same prepared state. Correctness checks use independently
reconstructed results after the response timer stops.
"""
import math
import time
import numpy as np
import pandas as pd
from scipy.special import expit

HROWS = 898933

def linear_score(features, model):
    return ((features - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']) + model['intercept']

def flat_model(model):
    weights = np.asarray(model['coef']) / np.asarray(model['scale'])
    return weights, float(model['intercept'] - weights @ np.asarray(model['mean']))


def load_snapshot(client, case, scale):
    """Three equal serving reads, followed by dense task-state reconstruction."""
    names = case['tables'][scale]
    start = time.perf_counter()
    mapping = client.query(
        f"SELECT replica,qid,user_id,course_id FROM {names['mapping']} "
        "ORDER BY replica,qid").result_rows
    catrows = client.query(
        f"SELECT replica,course_id,bid,n,s FROM {names['catalog']} "
        "ORDER BY replica,course_id,bid").result_rows
    values = client.query(
        f"SELECT replica,qid,bid,n,s FROM {names['aggregate']}").result_rows
    read_seconds = time.perf_counter() - start
    assert len(mapping) == len(case['cases']) * scale
    assert all((r[0], r[1]) == (i // len(case['cases']), i % len(case['cases']))
               for i, r in enumerate(mapping))
    catbase = pd.DataFrame([r[1:] for r in catrows if r[0] == 0],
                           columns=['course_id', 'bid', 'n', 's'])
    q = structure(case['cases'], catbase, scale)
    counts = np.zeros(q['basecells'] * scale, dtype=np.int64)
    sums = np.zeros_like(counts)
    arr = np.asarray(values, dtype=np.int64).reshape(-1, 5)
    positions = arr[:, 0] * q['basecells'] + q['baseoff'][arr[:, 1]] + arr[:, 2]
    counts[positions], sums[positions] = arr[:, 3], arr[:, 4]
    q.update(N=counts, S=sums,
             catN=np.array([r[3] for r in catrows], dtype=np.int64),
             catS=np.array([r[4] for r in catrows], dtype=np.int64),
             scale=scale, table=names['history'],
             g_read_seconds=read_seconds, g_read_queries=3,
             returned_g_rows=len(mapping) + len(catrows) + len(values))
    return q

class Reader:
    def __init__(self,client,table):
        self.client=client;self.table=table;self.values=0;self.queries=0
        self.engine_rows=0;self.engine_bytes=0;self.seconds=0.
    def read(self,ids):
        start=time.perf_counter();out={};ids=np.unique(ids)
        for lo in range(0,len(ids),4096):
            chunk=ids[lo:lo+4096]
            result=self.client.query('SELECT eid,is_correct FROM '+self.table+' WHERE eid IN ('+','.join(map(str,chunk))+')',
                settings={'log_queries':1,'log_comment':'gform_maintenance'})
            out.update((int(k),int(v)) for k,v in result.result_rows)
            self.queries+=1
            self.engine_rows+=int(result.summary.get('read_rows',0))
            self.engine_bytes+=int(result.summary.get('read_bytes',0))
        assert len(out)==len(ids)
        self.values+=len(ids);self.seconds+=time.perf_counter()-start
        return out
    def stats(self):return dict(retrieved_values=self.values,history_queries=self.queries,
        engine_read_rows=self.engine_rows,engine_read_bytes=self.engine_bytes,history_seconds=self.seconds)

def structure(cases,cat,scale):
    baseoff=np.r_[0,np.cumsum([len(c['ids']) for c in cases])[:-1]].astype(np.int64)
    basecells=sum(len(c['ids']) for c in cases);n=len(cases)
    cat=cat.sort_values(['course_id','bid']).reset_index(drop=True)
    lookup={(r.course_id,int(r.bid)):i for i,r in cat.iterrows()}
    ci=np.concatenate([np.array([lookup[(c['key'][1],j)] for j in range(len(c['ids']))],dtype=np.int64) for c in cases])
    off=np.tile(baseoff,scale)+np.repeat(np.arange(scale)*basecells,n)
    target=off+np.tile([c['target'] for c in cases],scale)
    return dict(off=off,ends=np.r_[off[1:],basecells*scale],target=target,n=n*scale,nbase=n,
        baseoff=baseoff,basecells=basecells,ncats=len(cat)*scale,
        ci=np.tile(ci,scale)+np.repeat(np.arange(scale)*len(cat),basecells),
        cellq=np.repeat(np.arange(n*scale),np.tile([len(c['ids']) for c in cases],scale)))

def expanded_journal(case,q):
    j=case['journal'];scale=q['scale'];m=len(j)
    return dict(eid=np.tile(j.eid.to_numpy(np.int64),scale)+np.repeat(np.arange(scale)*HROWS,m),
        gid=np.tile(q['baseoff'][j.qid]+j.bid.to_numpy(np.int64),scale)+np.repeat(np.arange(scale)*q['basecells'],m),
        new=np.tile(j['new'].to_numpy(np.int64),scale))

def state(q,j):
    m=np.bincount(j['gid'],minlength=len(q['S'])).astype(np.int64)
    new=np.bincount(j['gid'],weights=j['new'],minlength=len(q['S'])).astype(np.int64)
    assert np.all(m<=q['N'])
    return dict(changed=m,newsum=new,known=np.zeros_like(m),unknown=m.copy(),mask=np.ones(len(j['eid']),bool))

def bounds(q,st):
    rest=q['S']-st['known'];untouched=q['N']-st['changed']
    low=np.maximum(0,rest-untouched);high=np.minimum(st['unknown'],rest)
    fixed=q['S']+st['newsum']-st['known']
    L=fixed-high;U=fixed-low
    assert np.all(L<=U) and np.all(L>=0) and np.all(U<=q['N'])
    return L,U

def refine(q,j,st,ix,reader):
    old=reader.read(j['eid'][ix])
    v=np.fromiter((old[int(k)] for k in j['eid'][ix]),dtype=np.int64,count=len(ix))
    np.add.at(st['known'],j['gid'][ix],v)
    np.add.at(st['unknown'],j['gid'][ix],-1)
    st['mask'][ix]=False

def answer(q,S,models):
    catS=q['catS']+np.bincount(q['ci'],weights=S-q['S'],minlength=q['ncats'])
    ci=q['ci'];off=q['off'];target=q['target']
    Ntot=np.add.reduceat(q['N'],off);Stot=np.add.reduceat(S,off)
    static=np.column_stack((np.log1p(q['catN'][ci]),(catS[ci]+1)/(q['catN'][ci]+2)))
    Xc=np.column_stack((Ntot,Stot,q['N'][target],S[target],static[target]))
    Xr=np.column_stack((q['N'],static))
    p=expit(linear_score(Xc, models['classifier']))
    score=linear_score(Xr, models['ranker'])
    order=np.array([np.lexsort((np.arange(t-s),-score[s:t]))[:5] for s,t in zip(off,q['ends'])])
    return dict(p=p,cl=p>=.5,r=order,mon=np.column_stack((Ntot,Stot)),rank_scores=score)

def certify(q,L,U,models):
    """Certify classification and ranking from aggregate bounds.

    Shared target contributions are combined before taking absolute values.
    Decisions near a floating-point boundary require further refinement.
    """
    mid=(L+U)*.5;rad=(U-L)*.5;off=q['off'];target=q['target'];ci=q['ci']
    out=answer(q,mid,models)
    wc,bc=flat_model(models['classifier']);wr,br=flat_model(models['ranker'])
    cm=q['catS']+np.bincount(ci,weights=mid-q['S'],minlength=q['ncats'])
    cr=np.bincount(ci,weights=rad,minlength=q['ncats'])
    st=np.add.reduceat(mid,off);sr=np.add.reduceat(rad,off);nt=np.add.reduceat(q['N'],off)
    tc=ci[target];k=wc[5]/(q['catN'][tc]+2)
    score=bc+wc[0]*nt+wc[1]*st+wc[2]*q['N'][target]+wc[3]*mid[target]+wc[4]*np.log1p(q['catN'][tc])+wc[5]*(cm[tc]+1)/(q['catN'][tc]+2)
    radius=abs(wc[1])*(sr-rad[target])+np.abs(wc[1]+wc[3]+k)*rad[target]+np.abs(k)*(cr[tc]-rad[target])
    guard=1e-10*(1+abs(bc)+abs(wc[0])*nt+abs(wc[1])*st+abs(wc[2])*q['N'][target]+abs(wc[3])*mid[target]+abs(wc[4])*np.log1p(q['catN'][tc])+abs(wc[5]))
    class_ok=(score-radius>guard)|(score+radius < -guard)
    local_exact=sr==0
    class_ok|=local_exact & (cr[tc]==0)
    rank_radius=abs(wr[2])*cr[ci]/(q['catN'][ci]+2)
    selected=out['r']+off[:,None];scores=out['rank_scores']
    def uncertain_pair(i,j):
        pair_radius=rank_radius[i]+rank_radius[j]
        error=1e-10*(1+np.abs(scores[i])+np.abs(scores[j])+pair_radius)
        return (pair_radius>0)&(scores[i]-scores[j]-pair_radius<=error)
    inside_bad=uncertain_pair(selected[:,:-1],selected[:,1:]).any(axis=1)
    is_selected=np.zeros(len(q['N']),bool);is_selected[selected.ravel()]=True
    outside_bad=uncertain_pair(selected[:,-1][q['cellq']],np.arange(len(q['N'])))&~is_selected
    rank_ok=~inside_bad & (np.bincount(q['cellq'],weights=outside_bad,minlength=q['n'])==0)
    return out,class_ok,rank_ok

def required_groups(q,st,L,U,cl_ok,rank_ok):
    need=(~cl_ok)[q['cellq']]
    cats=np.zeros(q['ncats'],bool)
    cats[q['ci'][q['target'][~cl_ok]]]=True
    cats[q['ci'][(~rank_ok)[q['cellq']]]]=True
    need|=cats[q['ci']]
    need&=L<U
    return need

def read_estimate(profile,m):
    return math.ceil(m/4096)*profile['read_fixed']+m*profile['read_per_value'] if m else 0.

def service(client,case,scale,models,contract,method,profile,complete=True):
    start=time.perf_counter();q=load_snapshot(client,case,scale);j=expanded_journal(case,q)
    rd=Reader(client,q['table']);checks=passes=0;path=method
    dispatch_start=time.perf_counter()
    if method=='adaptive':
        predicted_check=profile['check_per_case']*q['n']
        predicted_read=read_estimate(profile,len(j['eid']))
        if contract!='decisions' or predicted_check>=.5*predicted_read:path='full_update'
        else:path='selective'
    dispatch=time.perf_counter()-dispatch_start
    st=None;L=U=None;check_time=0.;fallback=0
    if path=='full_update':
        old=rd.read(j['eid']);S=q['S'].copy()
        vv=np.fromiter((old[int(k)] for k in j['eid']),np.int64,count=len(j['eid']))
        np.add.at(S,j['gid'],j['new']-vv);out=answer(q,S,models)
        L=U=S
    else:
        st=state(q,j);L,U=bounds(q,st)
        if path=='bounds_batch' or contract!='decisions':
            ix=np.flatnonzero(st['mask']&(L<U)[j['gid']])
            if len(ix):refine(q,j,st,ix,rd)
            L,U=bounds(q,st);assert np.array_equal(L,U)
            out=answer(q,L,models)
            path='bounds_batch' if path=='bounds_batch' else 'selective_exact'
        else:
            while True:
                if np.array_equal(L,U):out=answer(q,L,models);break
                ct=time.perf_counter();out,cl_ok,rk_ok=certify(q,L,U,models);check_time+=time.perf_counter()-ct
                checks+=q['n'];passes+=1
                if cl_ok.all() and rk_ok.all():break
                need=required_groups(q,st,L,U,cl_ok,rk_ok)
                ix=np.flatnonzero(st['mask']&need[j['gid']]);assert len(ix)
                # One batch combines requests from all tasks and participations.
                chunk=4*2**(passes-1)
                order=np.lexsort((j['eid'][ix],j['gid'][ix]));ix=ix[order]
                groups=j['gid'][ix];first=np.r_[0,np.flatnonzero(np.diff(groups))+1]
                local=np.arange(len(ix))-np.repeat(first,np.diff(np.r_[first,len(ix)]))
                take=ix[local<chunk]
                stop=passes>=3
                if method=='adaptive':
                    per_check=max(profile['check_per_case'],check_time/(passes*q['n']))
                    stop|=per_check*q['n']+read_estimate(profile,len(take))>=.8*read_estimate(profile,len(ix))
                if stop:
                    take=np.flatnonzero(st['mask']&(L<U)[j['gid']]);fallback+=1
                refine(q,j,st,take,rd);L,U=bounds(q,st)
                # Fallback resolves every uncertain group before returning an exact answer.
                if stop:
                    L,U=bounds(q,st);assert np.array_equal(L,U)
                    out=answer(q,L,models);break
    elapsed=time.perf_counter()-start
    first_stats=rd.stats()
    unresolved=0 if st is None else int(np.count_nonzero(st['mask']&(L<U)[j['gid']]))
    state_bytes=0 if st is None else int(sum(v.nbytes for v in st.values())+L.nbytes+U.nbytes)
    result=dict(scale=scale,cut=case['cut'],contract=contract,method=method,selected_path=path,
        n=q['n'],journal_records=len(j['eid']),service_seconds=elapsed,check_seconds=check_time,
        g_read_seconds=q['g_read_seconds'],g_read_queries=3,checks=checks,check_passes=passes,
        fallbacks=fallback,dispatch_seconds=dispatch,unresolved_values=unresolved,
        unresolved_cells=int(np.count_nonzero(L<U)),extra_state_array_bytes=state_bytes,**first_stats)
    # Independently reconstructed updated outputs; no oracle reaches strategy code.
    reference=case['updated'];refp=np.tile(reference['d28_NS']['p'],scale)
    refr=np.tile(np.array(reference['d28_N']['r']),(scale,1));refmon=np.tile(case['updated_monitor'],(scale,1))
    assert np.array_equal(out['cl'],refp>=.5),(contract,method,'class')
    assert np.array_equal(out['r'],refr),(contract,method,'rank')
    if contract!='decisions':assert np.array_equal(out['mon'],refmon),(contract,method,'monitor')
    prob_error=float(np.max(np.abs(out['p']-refp))) if contract=='numeric_monitor' else None
    if prob_error is not None:assert prob_error<=1e-12,prob_error
    result.update(mismatches=0,max_probability_error=prob_error)
    finish=0.;finish_values=0
    if complete and unresolved:
        t=time.perf_counter();ix=np.flatnonzero(st['mask']&(L<U)[j['gid']]);refine(q,j,st,ix,rd)
        L,U=bounds(q,st);assert np.array_equal(L,U);final=answer(q,L,models);finish=time.perf_counter()-t
        finish_values=rd.values-first_stats['retrieved_values']
        assert np.array_equal(final['mon'],refmon) and np.array_equal(final['r'],refr)
        assert np.max(np.abs(final['p']-refp))<=1e-12
    result.update(completion_seconds=finish,completion_values=finish_values,
        service_plus_completion_seconds=elapsed+finish,total_history_values=rd.values)
    return result
