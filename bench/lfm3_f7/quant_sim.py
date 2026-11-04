import sys, json, numpy as np
sys.path.insert(0,'bench/lfm3_f6'); sys.path.insert(0,'bench/lfm3_f5')
import os; os.environ.setdefault('LFM3_TOKENIZER','/tmp/f4_base/lfm3-1.7b-data2x-ep4-merged/tokenizer.json')
from e2e_build_f6 import Builder, OP
b=Builder()
P=[json.loads(l) for l in open('bench/lfm3_f6/data/e2e/prompts_recorte.jsonl')]
P=[p for p in P if p['cell']=='op_d50']
V=b.comp.vecs.copy()
def q8(v):
    s=np.abs(v).max(1,keepdims=True)/127.0
    return (np.round(v/s).astype(np.int8).astype(np.float32)*s).astype(np.float32)
def f16(v): return v.astype(np.float16).astype(np.float32)
for name,fn in [('fp32',lambda v:v),('fp16',f16),('int8',q8)]:
    b.comp.vecs=fn(V)
    same=0
    for p in P:
        q=p['messages'][1]['content']
        c=b.cell(q,OP,'c5')
        same+= c['text']==p['messages'][-1]['content']
    print(name,same,len(P),flush=True)
