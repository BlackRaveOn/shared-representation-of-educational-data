"""Native aggregate storage and measurement boundaries for all datasets."""
import hashlib
import time
import uuid

import numpy as np
from .runtime import dumps


class Native:
    def __init__(self, client):
        self.c = client
        self.db = 'gform_local_' + uuid.uuid4().hex[:12]
        self.owned = set()
        self.views = set()
        self.generation = 0
        assert not client.query(f'EXISTS DATABASE {self.db}').first_row[0]
        client.command(f'CREATE DATABASE {self.db} ENGINE=Atomic')

    def full(self, name):
        assert name.replace('_','').isalnum()
        return self.db+'.'+name

    def create(self, name, columns, order):
        if name in self.owned:
            self.c.command(f'DROP TABLE IF EXISTS {self.full(name)} SYNC')
        else:
            assert not self.c.query(f'EXISTS TABLE {self.full(name)}').first_row[0]
            self.owned.add(name)
        self.c.command(f'CREATE TABLE {self.full(name)} ({columns}) ENGINE=MergeTree ORDER BY {order} SETTINGS index_granularity=512,index_granularity_bytes=0,min_rows_for_wide_part=0,min_bytes_for_wide_part=0')

    def compact(self, name):
        self.c.command(f'OPTIMIZE TABLE {self.full(name)} FINAL')

    def load(self, name, data):
        columns=['eid','qid','item','day','value']
        self.create(name,'eid UInt64 CODEC(ZSTD(3)),qid UInt32 CODEC(ZSTD(3)),item UInt32 CODEC(ZSTD(3)),day UInt8 CODEC(ZSTD(3)),value UInt8 CODEC(ZSTD(3))','eid')
        for offset in range(0,len(data),250000):
            self.c.insert_df(self.full(name),data.iloc[offset:offset+250000][columns])
        self.compact(name)

    def build(self, prefix, source, specs):
        self.generation += 1
        prefix += '_g' + str(self.generation)
        tables=[]
        for spec in specs:
            name=prefix+'_s'+str(spec['slot'])
            fields=['qid UInt32 CODEC(ZSTD(3))']
            if spec['by_item']: fields.append('item UInt32 CODEC(ZSTD(3))')
            fields+=['bin UInt8 CODEC(ZSTD(3))','n UInt64 CODEC(ZSTD(3))']
            if spec['with_sum']:fields.append('s UInt64 CODEC(ZSTD(3))')
            keys=['qid']+(['item'] if spec['by_item'] else [])
            self.create(name,','.join(fields),'('+','.join(keys+['bin'])+')')
            dayexpr=f'intDiv(day,{int(spec["days"])})'
            selects=keys+[f'toUInt8({dayexpr})','count()']+(['sum(value)'] if spec['with_sum'] else [])
            self.c.command(f'INSERT INTO {self.full(name)} SELECT '+','.join(selects)+f' FROM {self.full(source)} GROUP BY '+','.join(keys+[dayexpr]))
            self.compact(name)
            tables.append({'name':name,'spec':spec})
        return {'tables':tables,'specs':specs,'generation':self.generation}

    def query(self, plan):
        parts=[]
        for t in plan['tables']:
            s=t['spec']
            item='toInt64(item)' if s['by_item'] else 'toInt64(-1)'
            value='s' if s['with_sum'] else 'toUInt64(0)'
            parts.append(f'SELECT toInt64({s["slot"]}) AS slot,toInt64(qid) AS qid,{item} AS item,toInt64(bin) AS bin,n,{value} AS s FROM {self.full(t["name"])}')
        return ' UNION ALL '.join(parts)

    def publish(self, name, plan):
        self.c.command(f'CREATE OR REPLACE VIEW {self.full(name)} AS '+self.query(plan))
        self.views.add(name)
        return name

    def answer(self, snapshot, plan, shared, changed=False, view=None):
        start=time.perf_counter()
        query=f'SELECT * FROM {self.full(view)}' if view else self.query(plan)
        rows=np.asarray(self.c.query(query).result_rows,dtype=np.int64).reshape(-1,6)
        result=snapshot.evaluate(rows,shared=shared,changed=changed)
        return time.perf_counter()-start,result

    def table_bytes(self, names):
        quoted=','.join("'"+name+"'" for name in names)
        return int(self.c.query(f"SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database='{self.db}' AND table IN ({quoted})").first_row[0])

    def storage(self, snapshot, plan):
        a=self.table_bytes([t['name'] for t in plan['tables']])
        b=len(dumps(snapshot.model_payload).encode('utf8'))
        c=len(dumps(plan['specs']).encode('utf8'))
        return {'aggregate_bytes':a,'model_bytes':b,'plan_metadata_bytes':c,'total_bytes':a+b+c}

    def retire(self, plan):
        if plan is None:
            return
        for table in plan['tables']:
            name = table['name']
            assert name in self.owned
            self.c.command(f'DROP TABLE {self.full(name)} SYNC')
            self.owned.remove(name)

    def replace(self, prefix, source, specs, previous=None, view=None):
        """Create fresh tables, publish if needed, then synchronously retire old ones."""
        plan = self.build(prefix, source, specs)
        published = self.publish(view, plan) if view is not None else None
        self.retire(previous)
        return plan, published

    def verify_raw(self, name, data):
        """Exact row comparison and canonical hash without materializing a huge response."""
        columns = ['eid', 'qid', 'item', 'day', 'value']
        expected = data if data.eid.is_monotonic_increasing else data.sort_values('eid')
        digest = hashlib.sha256()
        cursor = 0
        query = f'SELECT {",".join(columns)} FROM {self.full(name)} ORDER BY eid'
        with self.c.query_row_block_stream(query) as stream:
            for block in stream:
                actual = np.asarray(block, dtype='<u8').reshape(-1, 5)
                reference = expected.iloc[cursor:cursor + len(actual)][columns].to_numpy(dtype=np.uint64)
                assert np.array_equal(actual, reference), name
                digest.update(actual.tobytes())
                cursor += len(actual)
        assert cursor == len(data), (name, cursor, len(data))
        return {'table': name, 'rows': len(data), 'ordered_rows_exact': True,
                'sha256': digest.hexdigest()}

    def close(self):
        assert self.db.startswith('gform_local_')
        self.c.command(f'DROP DATABASE {self.db} SYNC')


def compare(actual, expected):
    assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
    details={}
    for key in actual:
        a,b=np.asarray(actual[key]),np.asarray(expected[key])
        assert a.shape==b.shape,(key,a.shape,b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            assert np.array_equal(a,b),key
            details[key]={'shape':list(a.shape),'exact':True}
        else:
            diff=float(np.max(np.abs(a-b))) if a.size else 0.0
            assert diff<=1e-12,(key,diff)
            details[key]={'shape':list(a.shape),'max_abs_diff':diff}
    return details
