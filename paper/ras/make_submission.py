"""Assemble the files uploaded to the journal's submission system.

Builds paper/ras/submission/ with a flat layout: main.tex (figure paths rewritten),
refs.bib, main.bbl, each figure as its own file (Figure_1.pdf is the TikZ pipeline
exported standalone), highlights.txt and the compiled main.pdf; then compiles the
folder from scratch to confirm it builds on its own, and zips it.

Run from paper/ras after building main.pdf:  python3 make_submission.py
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission"
FIGS = {"../figs/fig_failures.pdf": "Figure_2.pdf", "../figs/fig_scores.pdf": "Figure_3.pdf"}

FIG1 = r"""\documentclass[12pt,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\begin{document}
\setlength{\textwidth}{390pt}
%s
\end{document}
"""


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tex = (HERE / "main.tex").read_text()

    # Figure 1: the TikZ picture, with the equation reference resolved from main.aux
    eq = re.search(r"\\newlabel\{eq:score\}\{\{(\d+)\}", (HERE / "main.aux").read_text()).group(1)
    tikz = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S).group(0)
    work = OUT / "_fig1"
    work.mkdir()
    (work / "Figure_1.tex").write_text(FIG1 % tikz.replace(r"\ref{eq:score}", eq))
    run(["pdflatex", "-interaction=nonstopmode", "Figure_1.tex"], work)
    shutil.copy(work / "Figure_1.pdf", OUT / "Figure_1.pdf")
    shutil.rmtree(work)

    for src, dst in FIGS.items():
        assert tex.count("{" + src + "}") == 1, src
        tex = tex.replace("{" + src + "}", "{" + dst + "}")
        shutil.copy(HERE / src, OUT / dst)
    assert tex.count(r"\bibliography{../refs}") == 1
    tex = tex.replace(r"\bibliography{../refs}", r"\bibliography{refs}")
    (OUT / "main.tex").write_text(tex)
    shutil.copy(HERE.parent / "refs.bib", OUT / "refs.bib")
    shutil.copy(HERE / "highlights.txt", OUT / "highlights.txt")

    # compile the flat folder from scratch in a copy, so only source files are uploaded
    check = HERE / "_submission_check"
    if check.exists():
        shutil.rmtree(check)
    shutil.copytree(OUT, check)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
        run(cmd, check)
    log = (check / "main.log").read_text(errors="ignore")
    assert not re.search(r"(Reference|Citation) `[^']+' .*undefined", log), "undefined references in the submission build"
    shutil.copy(check / "main.bbl", OUT / "main.bbl")
    shutil.copy(check / "main.pdf", OUT / "main.pdf")
    shutil.rmtree(check)

    shutil.make_archive(str(HERE / "submission"), "zip", OUT)
    print("wrote", OUT, "and", HERE / "submission.zip")
    for f in sorted(OUT.iterdir()):
        print(" ", f.name)


if __name__ == "__main__":
    main()
