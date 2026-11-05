// Harness de desktop que replica o nativeGenerate do llama_jni.cpp (decode em fatias de 512,
// cadeia de samplers, accept do prompt, reuso de prefixo) para validar (1) reuso de prefixo
// == decode do zero e (2) greedy determinístico. Uso: jni_harness model.gguf prompts.txt
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

struct Eng { llama_model* m; llama_context* c; std::vector<llama_token> cached; };

static std::vector<llama_token> tok(const llama_vocab* v, const std::string& s) {
    int n = -llama_tokenize(v, s.c_str(), s.size(), nullptr, 0, false, true);
    std::vector<llama_token> t(n);
    llama_tokenize(v, s.c_str(), s.size(), t.data(), n, false, true);
    return t;
}

static std::string gen(Eng& e, const std::string& prompt, int maxt, bool keep, float temp, int* reused) {
    const llama_vocab* v = llama_model_get_vocab(e.m);
    auto pt = tok(v, prompt);
    llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_penalties(llama_vocab_n_tokens(v), 64, 1.1f, 0, 0));
    llama_sampler_chain_add(s, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(s, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(s, llama_sampler_init_dist(42));
    size_t n_past = 0;
    if (keep && !e.cached.empty() && e.cached.size() < pt.size() && std::equal(e.cached.begin(), e.cached.end(), pt.begin())) {
        n_past = e.cached.size();
    } else { llama_memory_clear(llama_get_memory(e.c), false); e.cached.clear(); }
    *reused = (int) n_past;
    for (size_t i = n_past; i < pt.size(); i += 512) {
        int cs = std::min((size_t)512, pt.size() - i);
        llama_batch b = llama_batch_init(cs, 0, 1);
        for (int j = 0; j < cs; j++) { b.token[j]=pt[i+j]; b.pos[j]=i+j; b.n_seq_id[j]=1; b.seq_id[j][0]=0; b.logits[j]=(i+j==pt.size()-1); }
        b.n_tokens = cs;
        if (llama_decode(e.c, b)) { fprintf(stderr,"decode fail\n"); }
        llama_batch_free(b);
    }
    e.cached = pt;
    for (auto t : pt) llama_sampler_accept(s, t);
    std::string out;
    for (int k = 0; k < maxt; k++) {
        llama_token t = llama_sampler_sample(s, e.c, -1);
        if (llama_vocab_is_eog(v, t)) break;
        char buf[256]; int n = llama_token_to_piece(v, t, buf, sizeof buf, 0, true);
        out.append(buf, n);
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0]=t; b.pos[0]=e.cached.size(); b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=1; b.n_tokens=1;
        llama_decode(e.c, b); llama_batch_free(b);
        e.cached.push_back(t);
    }
    llama_sampler_free(s);
    return out;
}

int main(int argc, char** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params(); cp.n_ctx=4096; cp.n_batch=512; cp.n_ubatch=512; cp.n_threads=8; cp.n_threads_batch=8;
    llama_context* c = llama_init_from_model(m, cp);
    Eng e{m, c, {}};
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string all = ss.str();
    const std::string SEP = "\n=====SEP=====\n";
    size_t pos = 0; int idx = 0, same = 0, total = 0;
    while (pos < all.size()) {
        size_t nx = all.find(SEP, pos); if (nx == std::string::npos) nx = all.size();
        std::string syn = all.substr(pos, nx - pos); pos = nx + SEP.size();
        if (syn.empty()) continue;
        std::string marker = "<|im_start|>assistant\n<think></think><tool_call>";
        size_t k = syn.find(marker);
        std::string dec = syn.substr(0, k) + "<|im_start|>assistant\n<think></think>";
        int r;
        std::string a = gen(e, syn, 120, false, 0.0f, &r);                // do zero
        gen(e, dec, 0, false, 0.0f, &r);                                   // decisão (0 tokens gerados)
        std::string b = gen(e, syn, 120, true, 0.0f, &r);                 // reuso de prefixo
        int reused = r;
        std::string a2 = gen(e, syn, 120, false, 0.0f, &r);               // do zero de novo (determinismo)
        total++; same += (a == b);
        printf("[%d] prefix_reused=%d  zero==prefix:%d  zero==zero:%d\n  A=%s\n", idx++, reused, a==b, a==a2, a.substr(0,160).c_str());
        if (a != b) printf("  B=%s\n", b.substr(0,160).c_str());
        fflush(stdout);
    }
    printf("SAME %d/%d\n", same, total);
}
