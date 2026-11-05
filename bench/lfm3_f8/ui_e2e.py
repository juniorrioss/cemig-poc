#!/usr/bin/env python3
"""F8 — E2E das 8 perguntas-exemplo PELA UI no S24+ (adb + uiautomator).

Para cada chip: Nova conversa (confirmação) -> rola até o chip -> toca -> espera o fim
(logcat "MainViewModel: ... Done" não existe; usa estado da UI: some a bolha "Pensando" e o
status volta a "Pronto") -> dump da UI -> extrai resposta, fontes (selo + item) e tempos.
Screenshot de cada resposta em screenshots/e2e_XX.png. Saída: data/ui_e2e_results.json.
Uso: python3 bench/lfm3_f8/ui_e2e.py <serial>
"""
import json, os, re, subprocess, sys, time, xml.etree.ElementTree as ET

SERIAL = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
ADB = ["adb", "-s", SERIAL]
QS = json.load(open(os.path.join(HERE, "../../android/app/src/main/assets/example_questions.json")))


def sh(*a, out=False):
    r = subprocess.run(ADB + ["shell", *a], capture_output=True, text=True)
    return r.stdout if out else None


def dump():
    sh("uiautomator", "dump", "/sdcard/f8ui.xml")
    x = subprocess.run(ADB + ["exec-out", "cat", "/sdcard/f8ui.xml"], capture_output=True, text=True).stdout
    return ET.fromstring(x[x.index("<?xml"):] if "<?xml" in x else x)


def nodes(root):
    for n in root.iter("node"):
        b = re.findall(r"\d+", n.get("bounds", "0,0,0,0"))
        yield n.get("text", ""), n.get("content-desc", ""), [int(v) for v in b]


def center(b):
    return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2


def tap(x, y):
    sh("input", "tap", str(x), str(y))


def screenshot(name):
    with open(os.path.join(HERE, "screenshots", name), "wb") as f:
        f.write(subprocess.run(ADB + ["exec-out", "screencap", "-p"], capture_output=True).stdout)


def new_conversation():
    for t, d, b in nodes(dump()):
        if d == "Nova conversa":
            tap(*center(b)); time.sleep(0.8)
            for t2, d2, b2 in nodes(dump()):
                if t2 == "Reiniciar":
                    tap(*center(b2)); time.sleep(0.8)
            return


def find_chip(q):
    for _ in range(8):
        for t, d, b in nodes(dump()):
            if t == q:
                return b
        sh("input", "swipe", "700", "2200", "700", "1400", "300"); time.sleep(0.6)
    return None


def wait_done(timeout=120):
    t0 = time.time()
    time.sleep(3)
    while time.time() - t0 < timeout:
        texts = [t for t, _, _ in nodes(dump())]
        if any(t.startswith("Pronto") for t in texts) and not any("Pensando" == t for t in texts):
            return time.time() - t0
        time.sleep(1.5)
    return None


results = []
for i, q in enumerate(QS, 1):
    new_conversation()
    b = find_chip(q["question"])
    if b is None:
        results.append({"id": q["id"], "error": "chip não encontrado"}); continue
    subprocess.run(ADB + ["logcat", "-c"])
    tap(*center(b))
    wall = wait_done()
    time.sleep(1)
    screenshot(f"e2e_{i:02d}.png")
    log = subprocess.run(ADB + ["logcat", "-d", "-s", "MainViewModel:I"], capture_output=True, text=True).stdout
    called = "buscar=true" in log
    # acumula textos rolando a conversa (o dump só traz o que está na tela)
    texts = []
    for k in range(6):
        for t, _, _ in nodes(dump()):
            if t and (not texts or t not in texts[-40:]):
                texts.append(t)
        sh("input", "swipe", "700", "2100", "700", "1100", "300"); time.sleep(0.7)
    screenshot(f"e2e_{i:02d}_fontes.png")
    # resposta = maior texto que não é a pergunta nem cabeçalho
    cand = [t for t in texts if t != q["question"] and len(t) > 40 and not t.startswith(("Segure", "Fale"))]
    answer = max(cand, key=len) if cand else ""
    # fontes visíveis: selos NR-xx / MANUAL NR-xx seguidos do item
    srcs = []
    for j, t in enumerate(texts):
        if re.fullmatch(r"(MANUAL )?NR-\d+", t) and j + 1 < len(texts):
            srcs.append(f"{t} {texts[j + 1]}")
    timing = next((t for t in texts if "tok/s" in t), "")
    has_sources_card = any(t.startswith("Fontes consultadas") for t in texts)
    rec = dict(n=i, id=q["id"], kind=q["kind"], expected=q["source"], question=q["question"],
               tool_called=called, sources_card=has_sources_card, sources_visible=srcs,
               expected_source_shown=any(q["source"].split(" ·")[0].upper() in s and (q["kind"] != "manual" or s.startswith("MANUAL")) for s in srcs),
               answer=answer, timing_line=timing, ui_wall_s=round(wall, 1) if wall else None)
    print(json.dumps(rec, ensure_ascii=False)[:400], flush=True)
    results.append(rec)

json.dump(results, open(os.path.join(HERE, "data/ui_e2e_results.json"), "w"), ensure_ascii=False, indent=1)
print("ok", sum(r.get("sources_card", False) for r in results), "/", len(results), "com fontes")
