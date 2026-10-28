"""Sanidade: carregar LFM3-1.7B bf16, aplicar chat template, gerar, conferir EOS."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM3-1.7B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
msgs = [{"role": "system", "content": "Você é um assistente de segurança do trabalho. Responda em português do Brasil, de forma curta."},
        {"role": "user", "content": "O que significa a sigla EPI?"}]
prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print("=== PROMPT RENDERIZADO (byte a byte) ===")
print(repr(prompt_text))
enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
ids = enc["input_ids"].to("cuda")
out = model.generate(ids, max_new_tokens=120, do_sample=True, temperature=0.1,
                     top_k=50, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
gen = out[0][ids.shape[1]:]
print("=== GEN TOKENS ===", len(gen))
print("=== RESPOSTA ===")
print(tok.decode(gen, skip_special_tokens=True))
print("EOS id final:", gen[-1].item(), "(esperado 124900 <|im_end|> se parou sozinho)")
assert gen[-1].item() == 124900, "EOS nao respeitado"
print("SANITY_OK")
