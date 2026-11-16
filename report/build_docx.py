"""Build report.docx from report.md (small Markdown subset) for pasting into the Apart template.

Supported: # title, ## / ### headings, paragraphs, '- ' bullets, numbered lines (kept as text),
pipe tables, ![caption](path) figures, **bold**, *italic*, `code`.
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
SRC = HERE / "report.md"
OUT = HERE / "report.docx"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_inline(par, text, size=None, italic=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2]); run.bold = True
        elif part.startswith("*"):
            run = par.add_run(part[1:-1]); run.italic = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1]); run.font.name = "Courier New"
        else:
            run = par.add_run(part)
        if italic:
            run.italic = True
        if size:
            run.font.size = Pt(size)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # Column widths over the 6.5-inch text width: simulate word wrapping at 8 pt and hand out width
    # in 0.05-inch steps to whichever column most reduces the table's total number of lines.
    plain = lambda c: re.sub(r"[*`]", "", c)

    def n_lines(text, w, bold):
        cap, lines, cur = max(1, int((w - 0.15) * (15 if bold else 17))), 1, 0
        for word in plain(text).split():
            if cur and cur + 1 + len(word) <= cap:
                cur += 1 + len(word)
            else:
                lines, cur = lines + (1 if cur else 0), len(word)
                if cur > cap:  # a word split across lines reads badly: penalise heavily
                    lines += 10
        return lines

    def height(ws):
        return sum(max(n_lines(c, ws[j], i == 0) for j, c in enumerate(r)) for i, r in enumerate(cells))

    ncol = len(cells[0])
    lens = [max(len(plain(r[j])) for r in cells[1:]) for j in range(ncol)]
    wts = [max(n, 8) ** 0.8 for n in lens]
    ws = [round(6.5 * w / sum(wts), 2) for w in wts]
    ws[0] += round(6.5 - sum(ws), 2)
    best = height(ws)
    improved = True
    while improved:  # local search: move width between any two columns, small or large steps
        improved = False
        for d in (1.0, 0.5, 0.2, 0.1, 0.05):
            for a_ in range(ncol):
                for b_ in range(ncol):
                    if a_ == b_ or ws[a_] - d < 0.45:
                        continue
                    trial = ws[:]
                    trial[a_] -= d
                    trial[b_] += d
                    h = height(trial)
                    if h < best:
                        ws, best, improved = trial, h, True
    widths = [Inches(w) for w in ws]
    for j, w in enumerate(widths):
        t.columns[j].width = w
    for i, r in enumerate(cells):
        for j, c in enumerate(r):
            t.cell(i, j).width = widths[j]
            p = t.cell(i, j).paragraphs[0]
            add_inline(p, f"**{c}**" if i == 0 and c else c, size=8)
    doc.add_paragraph()


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"; st.font.size = Pt(10)
    st.paragraph_format.space_after = Pt(4)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)
        s.top_margin = s.bottom_margin = Inches(0.9)

    lines = SRC.read_text().splitlines()
    i = 0
    table = []
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            table.append(line); i += 1
            if i == len(lines) or not lines[i].startswith("|"):
                add_table(doc, table); table = []
            continue
        if not line or line.lstrip().startswith("<!--"):
            i += 1; continue
        if line.startswith("# "):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(line[2:]); r.bold = True; r.font.size = Pt(15)
        elif line.startswith("## "):
            h = doc.add_heading(line[3:], level=1)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B); r.font.size = Pt(12.5)
        elif line.startswith("### "):
            h = doc.add_heading(line[4:], level=2)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B); r.font.size = Pt(11)
        elif m := re.match(r"!\[(.*)\]\((.*)\)", line):
            cap, path = m.groups()
            img = (HERE / path).resolve()
            width = {"fig1": 5.0, "fig2": 5.7, "fig3": 5.0, "figA1": 4.8, "figA2": 3.6}.get(img.name.split("_")[0], 5.0)
            doc.add_picture(str(img), width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph()
            add_inline(p, cap, size=8.5, italic=False)
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, line[2:])
        elif re.match(r"\d+\. ", line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            add_inline(p, line)
        else:
            p = doc.add_paragraph()
            first = line.startswith("**Juhong Park**") or line.startswith("*With Apart") or line.startswith("*Research conducted")
            if first:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, line)
        i += 1
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
