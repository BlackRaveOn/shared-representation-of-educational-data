"""Download public ALPACAS v1 and build auditable, outcome-free event inputs."""
from pathlib import Path
import json, hashlib, zipfile, argparse
import numpy as np
import pandas as pd
import requests

P = Path(__file__).resolve().parents[1]
def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def download(root):
    root.mkdir(parents=True, exist_ok=True)
    meta = requests.get('https://zenodo.org/api/records/17087849', timeout=60)
    meta.raise_for_status(); meta = meta.json(); save(root/'source_metadata.json', meta)
    for f in meta['files']:
        if f['key'] not in ['dataset.zip','course_info.json']: continue
        dest = root/f['key']
        if not dest.exists():
            with requests.get(f['links']['self'], stream=True, timeout=120) as r:
                r.raise_for_status()
                with dest.open('wb') as out:
                    for chunk in r.iter_content(1024*1024): out.write(chunk)
        assert hashlib.md5(dest.read_bytes()).hexdigest() == f['checksum'].split(':')[-1]
    with zipfile.ZipFile(root/'dataset.zip') as z:
        for i in z.infolist():
            if not i.filename.startswith('dataset/') or Path(i.filename).name.startswith('._'): continue
            dest=(root/'data'/i.filename).resolve()
            assert dest.is_relative_to((root/'data').resolve())
            z.extract(i,root/'data')

def prepare(root, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    info=json.loads((root/'course_info.json').read_text())
    audit={'source_doi':'10.5281/zenodo.17087849','dataset_zip_sha256':sha(root/'dataset.zip'),
           'course_info_sha256':sha(root/'course_info.json'),'years':{},
           'semantics':'LMS records are counted as records, not distinct intentional clicks. Forum contributions are counted by creation time. Post text, edit metadata, reply totals and final consumption totals are not features.',
           'identity':'Year, original course page and user identify a participation. No cross-year or cross-course person linkage is inferred.'}
    for year in ['1819','1920','2021']:
        src=root/'data'/'dataset'
        g=pd.read_excel(src/f'{year}_course_participation.xlsx')
        nsource=len(g); keys=['COURSE_ID','USER_ID']
        conflicts=g.groupby(keys).PASSED_FIRST_ATTEMPT.nunique()
        bad=set(conflicts[conflicts>1].index)
        g=g[[tuple(x) not in bad for x in g[keys].to_numpy()]].drop_duplicates(keys)
        assert g.PASSED_FIRST_ATTEMPT.isin([0,1]).all()
        g=g.sort_values(keys).reset_index(drop=True)
        g['qid']=np.arange(len(g),dtype='uint32')
        g['course']=g.COURSE_ID.str.replace(r' [12]$','',regex=True)
        g['course_flag']=(g.course=='Global economics').astype('uint8')
        lookup={(r.COURSE_ID,int(r.USER_ID)):int(r.qid) for r in g.itertuples()}
        parts=[]; counts={}; duplicate_ids={}
        for channel in [0,1]:
            if channel==0:
                chunks=pd.read_csv(src/f'{year}_log_activity.csv',usecols=['ACTION_ID','COURSE_ID','USER_ID','TIMESTAMP'],chunksize=250000)
                timecol='TIMESTAMP'; idcol='ACTION_ID'
            else:
                chunks=[pd.read_excel(src/f'{year}_df_contribution.xlsx')]
                timecol='DTCREATED'; idcol='POST_ID'
            totals={'raw_records':0,'unknown_or_excluded_participation':0,'outside_window':0,'invalid_time':0,'selected':0}
            offset=0; source_ids=[]
            for f in chunks:
                totals['raw_records']+=len(f)
                q=np.array([lookup.get((c,int(u)),-1) for c,u in f[keys].to_numpy()],dtype='int64')
                dt=pd.to_datetime(f[timecol],errors='coerce')
                starts=pd.to_datetime(f.COURSE_ID.str.replace(r' [12]$','',regex=True).map({k:pd.Timestamp(*v['course_start']) for k,v in info[year].items()}))
                days=(dt-starts).dt.total_seconds()/86400
                valid=(q>=0)&days.ge(0).to_numpy()&days.lt(56).to_numpy()
                totals['unknown_or_excluded_participation']+=int((q<0).sum())
                totals['invalid_time']+=int(dt.isna().sum())
                totals['outside_window']+=int(((q>=0)&~(days.ge(0)&days.lt(56)).to_numpy()).sum())
                totals['selected']+=int(valid.sum())
                # Original row identity, kept independent of potentially repeated source IDs.
                eid=np.arange(offset,offset+len(f),dtype='uint64')+channel*10**12
                parts.append(pd.DataFrame({'eid':eid[valid],'qid':q[valid].astype('uint32'),
                    'day':np.floor(days[valid]).astype('uint8').to_numpy(),'channel':np.full(valid.sum(),channel,dtype='uint8')}))
                source_ids.extend(zip(f.loc[valid,'COURSE_ID'],f.loc[valid,idcol]))
                offset+=len(f)
            duplicate_ids[str(channel)]=len(source_ids)-len(set(source_ids))
            counts[str(channel)]=totals
        events=pd.concat(parts,ignore_index=True)
        assert events.eid.is_unique
        # A source identity collision is an audit failure, not silently deduplicated.
        assert not any(duplicate_ids.values()), duplicate_ids
        events.to_parquet(out/f'events_{year}.parquet',index=False)
        g.rename(columns={'PASSED_FIRST_ATTEMPT':'y'})[['qid','COURSE_ID','USER_ID','course','course_flag','y']].to_parquet(out/f'participants_{year}.parquet',index=False)
        audit['years'][year]={'source_participation_rows':nsource,'served_participations':len(g),
            'conflicting_participation_keys_excluded':len(bad),'duplicate_keys_collapsed':nsource-len(g)-sum(tuple(x) in bad for x in pd.read_excel(src/f'{year}_course_participation.xlsx')[keys].to_numpy()),
            'courses':g.groupby('COURSE_ID').size().to_dict(),'counts':counts,
            'zero_activity_participations':int(len(g)-events.qid.nunique()),'source_id_collisions':duplicate_ids,
            'events_sha256':sha(out/f'events_{year}.parquet'),'participants_sha256':sha(out/f'participants_{year}.parquet')}
        print(year,json.dumps(audit['years'][year]),flush=True)
    save(out/'input_audit.json',audit)
    return audit


def prepare_alpacas(
    raw_dir: Path | str,
    out_dir: Path | str,
    *,
    download_source: bool = True,
) -> dict:
    """Prepare ALPACAS windows, downloading the pinned release when requested."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Use an empty output directory.')
    if download_source:
        download(raw_dir)
    return prepare(raw_dir, out_dir)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,default=P/'local_data/alpacas');ap.add_argument('--out-dir',type=Path,default=P/'outputs/prepared_alpacas');ap.add_argument('--existing',action='store_true');a=ap.parse_args()
    prepare_alpacas(a.raw_dir, a.out_dir, download_source=not a.existing)
