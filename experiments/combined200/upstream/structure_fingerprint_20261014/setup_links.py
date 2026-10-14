from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
for name in ['index.html','fingerprint.html','fingerprint_mix.html']:
    p=ROOT/name;s=p.read_text(encoding='utf-8')
    if 'href="structure_fingerprint.html"' not in s:
        s=s.replace('<main>','<main><nav class="toolbar"><a href="structure_fingerprint.html">果粒排布结构指纹样例 →</a></nav>',1)
    p.write_text(s,encoding='utf-8')
