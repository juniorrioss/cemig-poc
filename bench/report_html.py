#!/usr/bin/env python3
"""Gera bench/REPORT.html a partir de bench/REPORT.md (fonte canônica).

Conversor markdown→HTML mínimo em stdlib puro (sem pandoc/pip): cobre o
subconjunto usado no REPORT.md — headers, tabelas, listas (ordenadas e não,
com continuação indentada), blockquote, bold/italic/código inline, hr.
Uso: python3 bench/report_html.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "REPORT.md"
DST = HERE / "REPORT.html"

CSS = """
:root { --fg:#1a202c; --muted:#5a6572; --accent:#0b5d4f; --border:#d8dee5;
        --bg:#ffffff; --code:#f2f4f7; --th:#eef2f1; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f7f9; color:var(--fg);
       font:16px/1.65 Georgia, 'Times New Roman', serif; }
main { max-width: 980px; margin: 2rem auto; background:var(--bg);
       padding: 3rem 3.5rem; border:1px solid var(--border); border-radius:8px; }
h1 { font-size:1.9rem; line-height:1.25; border-bottom:3px solid var(--accent);
     padding-bottom:.5rem; }
h2 { font-size:1.45rem; margin-top:2.4rem; color:var(--accent);
     border-bottom:1px solid var(--border); padding-bottom:.3rem; }
h3 { font-size:1.15rem; margin-top:1.8rem; }
code { font:.85em/1.4 ui-monospace, 'Cascadia Code', Consolas, monospace;
       background:var(--code); padding:.1em .35em; border-radius:4px; }
blockquote { margin:1.2rem 0; padding:.7rem 1.2rem; border-left:4px solid var(--accent);
             background:#f4f8f7; color:var(--muted); font-size:.95em; }
table { border-collapse:collapse; width:100%; margin:1.1rem 0; font-size:.86em;
        font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; }
th, td { border:1px solid var(--border); padding:.42em .6em; text-align:left;
         vertical-align:top; }
th { background:var(--th); font-weight:600; }
tr:nth-child(even) td { background:#fafbfc; }
hr { border:none; border-top:1px solid var(--border); margin:2.2rem 0; }
ul, ol { padding-left:1.6rem; }
li { margin:.3rem 0; }
strong { color:#083f36; }
@media (max-width: 720px) { main { padding:1.2rem; } table { font-size:.75em; } }
"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"<em>\1</em>", text)
    return text


def table_block(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    if len(rows) < 2:
        return ""
    aligns = []
    for cell in rows[1]:
        if cell.endswith(":") and cell.startswith(":"):
            aligns.append("center")
        elif cell.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in rows[0]]
    out.append("</tr></thead><tbody>")
    for row in rows[2:]:
        out.append("<tr>")
        for i, c in enumerate(row):
            a = aligns[i] if i < len(aligns) else "left"
            style = f' style="text-align:{a}"' if a != "left" else ""
            out.append(f"<td{style}>{inline(c)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    in_ul = in_ol = in_bq = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_bq():
        nonlocal in_bq
        if in_bq:
            out.append("</blockquote>")
            in_bq = False

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_lists(); close_bq()
            tbl = [ln]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            out.append(table_block(tbl))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            close_lists(); close_bq()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if s in ("---", "***", "___"):
            close_lists(); close_bq()
            out.append("<hr>")
            i += 1
            continue

        if s.startswith(">"):
            close_lists()
            if not in_bq:
                out.append("<blockquote>"); in_bq = True
            body = s[1:].strip()
            out.append(f"<p>{inline(body)}</p>" if body else "")
            i += 1
            continue

        m = re.match(r"^(\s*)([-*])\s+(.*)$", ln)
        if m and not s.startswith("|"):
            close_bq()
            if in_ol:
                out.append("</ol>"); in_ol = False
            if not in_ul:
                out.append("<ul>"); in_ul = True
            item = [m.group(3)]
            i += 1
            # continuação indentada (inclui tabelas aninhadas)
            while i < len(lines):
                nxt = lines[i]
                if re.match(r"^\s*$", nxt):
                    if i + 1 < len(lines) and re.match(r"^\s{2,}\S", lines[i + 1]):
                        item.append(""); i += 1; continue
                    break
                if re.match(r"^\s*([-*]|\d+\.)\s+", nxt) or not re.match(r"^\s{2,}\S", nxt):
                    break
                item.append(nxt.strip()); i += 1
            # tabela dentro do item?
            tbl_start = next((k for k, t in enumerate(item) if t.startswith("|")), None)
            if tbl_start is not None:
                head = " ".join(t for t in item[:tbl_start] if t)
                tbl = [t for t in item[tbl_start:] if t.startswith("|")]
                out.append(f"<li>{inline(head)}{table_block(tbl)}</li>")
            else:
                out.append(f"<li>{inline(' '.join(t for t in item if t))}</li>")
            continue

        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", ln)
        if m:
            close_bq()
            if in_ul:
                out.append("</ul>"); in_ul = False
            if not in_ol:
                out.append("<ol>"); in_ol = True
            item = [m.group(3)]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if re.match(r"^\s*$", nxt) or re.match(r"^\s*([-*]|\d+\.)\s+", nxt) or not re.match(r"^\s{2,}\S", nxt):
                    break
                item.append(nxt.strip()); i += 1
            out.append(f"<li>{inline(' '.join(item))}</li>")
            continue

        if not s:
            close_lists(); close_bq()
            i += 1
            continue

        # parágrafo (junta linhas contíguas)
        close_lists(); close_bq()
        para = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "|", ">", "-", "*")) or re.match(r"^\d+\.\s", nxt):
                break
            para.append(nxt); i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")

    close_lists(); close_bq()
    return "\n".join(x for x in out if x)


def main():
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    title = "Relatório vivo — Exploração LFM3-1.7B (F0 → F4)"
    DST.write_text(
        "<!DOCTYPE html>\n<html lang=\"pt-BR\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"gerado: {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
