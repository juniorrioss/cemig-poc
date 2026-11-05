# probe_engine_leak.py — Spark: reproduz o vazamento de estado entre conversas/sessões na mesma Engine (LiteRT-LM #3165) que motivou "1 Engine por prompt" no gen_litert.py.
import litert_lm,json,sys
r=[json.loads(l) for l in open("prompts_op344_rendered.jsonl")]
T="{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for c in m.content %}{{ c.text }}{% endfor %}{% endif %}{% endfor %}"
M="lrt/f4-r32-ep2_mixed_int4_ekv2048.litertlm"
be=sys.argv[1]
B=lambda: litert_lm.Backend.GPU() if be=="gpu" else litert_lm.Backend.CPU(thread_count=6)
def conv(eng,p,n=96):
    c=eng.create_conversation(chat_template=T, sampler_config=litert_lm.SamplerConfig(top_k=1,top_p=1.0,temperature=0.1,seed=42), max_output_tokens=n)
    m=c.send_message({"role":"user","content":p}, repetition_penalty_config=litert_lm.RepetitionPenaltyConfig(repetition_penalty=1.1, window_size=64))
    tc=c.token_count; c.close(); return m.contents[0].text if m.contents else "", tc
def sess(eng,p,n=96):
    s=eng.create_session(apply_prompt_template=False,max_output_tokens=n,sampler_config=litert_lm.SamplerConfig(top_k=1,top_p=1.0,temperature=0.1,seed=42))
    s.run_prefill([p]); o=s.run_decode(); s.close(); return o.texts[0]
A=r[100]["prompt_text"][15:]; Bp=r[250]["prompt_text"][15:]
e1=litert_lm.Engine(M,backend=B(),max_num_tokens=2048,cache_dir="lrt")
fb,tc=conv(e1,Bp); print("FRESH_CONV",tc,r[250]["n_prompt"],repr(fb[:150]),flush=True)
e1.close()
e2=litert_lm.Engine(M,backend=B(),max_num_tokens=2048,cache_dir="lrt")
conv(e2,A); ab,_=conv(e2,Bp); print("AFTER_CONV_EQ",ab==fb,repr(ab[:150]),flush=True)
e2.close()
e3=litert_lm.Engine(M,backend=B(),max_num_tokens=2048,cache_dir="lrt")
fs=sess(e3,Bp); sess(e3,A); as_=sess(e3,Bp); print("SESS_REPEAT_EQ",fs==as_,repr(fs[:100]),"|",repr(as_[:100]),flush=True)
