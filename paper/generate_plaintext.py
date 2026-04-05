#!/usr/bin/env python3
"""Convert dynaclip_neurips2026.tex to readable plain text."""

import re
import sys
from pathlib import Path

INPUT = Path(__file__).parent / "dynaclip_neurips2026.tex"
OUTPUT = Path(__file__).parent / "dynaclip_neurips2026.txt"


def convert(tex: str) -> str:
    # 1. Strip preamble (everything before \begin{document})
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]

    # Strip everything after \end{document}
    m = re.search(r"\\end\{document\}", tex)
    if m:
        tex = tex[: m.start()]

    # Remove tikzpicture environments entirely (they produce garbage in plaintext)
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", tex, flags=re.DOTALL)

    # Remove algorithm/algorithmic environments (optional: keep if desired)
    # tex = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", "", tex, flags=re.DOTALL)

    # 12. Remove pure LaTeX comment lines (starting with %)
    tex = re.sub(r"(?m)^[ \t]*%.*$", "", tex)
    # Remove inline trailing comments
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)

    # 2. Replace \methodname{} with "DynaCLIP"
    tex = re.sub(r"\\methodname\b\{?\}?", "DynaCLIP", tex)

    # Custom short commands
    tex = re.sub(r"\\ie\b\{?\}?", "i.e.", tex)
    tex = re.sub(r"\\eg\b\{?\}?", "e.g.", tex)
    tex = re.sub(r"\\etal\b\{?\}?", "et al.", tex)
    tex = re.sub(r"\\wrt\b\{?\}?", "w.r.t.", tex)

    # 15. \section{Title} -> "# Title"
    tex = re.sub(r"\\section\*?\{([^}]*)\}", r"\n# \1\n", tex)

    # 16. \subsection{Title} -> "## Title"
    tex = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n## \1\n", tex)

    tex = re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"\n### \1\n", tex)

    # 14. \paragraph{Title.} -> "## Title"
    tex = re.sub(r"\\paragraph\*?\{([^}]*)\}", r"\n## \1\n", tex)

    # 3. Replace \textbf{...}, \emph{...}, \texttt{...}, \best{...}, \second{...},
    #    and similar wrappers with their contents (handle nested braces up to 2 levels)
    brace_content = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
    for cmd in ["textbf", "textit", "emph", "texttt", "textsc", "underline",
                 "best", "second", "mathbf", "mathrm", "mathit", "mathcal",
                 "boldsymbol", "operatorname"]:
        tex = re.sub(rf"\\{cmd}" + brace_content, r"\1", tex)

    # 4. Remove \cite{...} references
    tex = re.sub(r"~?\\cite\w*" + brace_content, "", tex)

    # 5. Replace \ref{...} with [REF]
    tex = re.sub(r"\\[Cc]ref" + brace_content, "[REF]", tex)
    tex = re.sub(r"\\ref" + brace_content, "[REF]", tex)

    # 6. Remove \label{...}
    tex = re.sub(r"\\label" + brace_content, "", tex)

    # 10. Handle \caption{...} — keep the text
    tex = re.sub(r"\\caption" + brace_content, r"\1", tex)

    # Remove \vspace{...}, \hspace{...}
    tex = re.sub(r"\\[vh]space\*?" + brace_content, "", tex)

    # Remove \resizebox and similar sizing commands
    tex = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{", "", tex)

    # Remove \includegraphics[...]{...}
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?" + brace_content, "", tex)

    # Remove \usepackage, \documentclass, etc. leftover
    tex = re.sub(r"\\(usepackage|documentclass|bibliographystyle|bibliography|input)(\[[^\]]*\])?" + brace_content, "", tex)

    # 7. Strip \begin{...} and \end{...} environment declarations
    tex = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?", "", tex)
    tex = re.sub(r"\\end\{[^}]*\}", "", tex)

    # 8. Remove table rules
    tex = re.sub(r"\\(midrule|toprule|bottomrule|hline|cline\{[^}]*\})\b", "", tex)

    # 9. Replace & with " | "
    tex = tex.replace("&", " | ")

    # 10. Remove \centering, \small, \maketitle, \newpage, etc.
    tex = re.sub(r"\\(centering|small|large|Large|LARGE|huge|Huge|footnotesize|scriptsize|tiny|normalsize|maketitle|newpage|clearpage|noindent|raggedright|raggedleft|appendix)\b", "", tex)

    # Remove \item with optional [...]
    tex = re.sub(r"\\item\s*(\[[^\]]*\])?\s*", "  * ", tex)

    # Remove remaining \command{...} where we want to keep the content
    # (catch-all for things like \text{...}, \mbox{...}, etc.)
    for cmd in ["text", "mbox", "textrm", "url", "href", "footnote", "footnotetext"]:
        tex = re.sub(rf"\\{cmd}" + brace_content, r"\1", tex)

    # Remove \newcommand, \definecolor, \renewcommand lines
    tex = re.sub(r"\\(newcommand|renewcommand|definecolor|DeclareMathOperator)\b.*", "", tex)

    # Remove TikZ and other complex commands
    tex = re.sub(r"\\(usetikzlibrary|tikzstyle|pgfmathsetmacro)\b.*", "", tex)

    # Remove remaining backslash commands with no arguments (like \centering etc.)
    # but be careful not to strip too much
    tex = re.sub(r"\\(quad|qquad|,|;|!|\\)", " ", tex)
    tex = re.sub(r"\\(left|right|big|Big|bigg|Bigg)([|().\[\]])", r"\2", tex)
    tex = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\b", "", tex)
    tex = re.sub(r"\\(dots|cdots|ldots|cdot|times)", "...", tex)
    tex = re.sub(r"\\(rightarrow|leftarrow|Rightarrow|Leftarrow|to)\b", "->", tex)
    tex = re.sub(r"\\(in|notin|subset|supset|cup|cap|setminus)\b", lambda m: {
        "in": "in", "notin": "not in", "subset": "subset", "supset": "supset",
        "cup": "U", "cap": "∩", "setminus": "\\"
    }.get(m.group(1), m.group(1)), tex)
    tex = re.sub(r"\\(leq|geq|neq|approx|sim|propto|equiv)\b", lambda m: {
        "leq": "<=", "geq": ">=", "neq": "!=", "approx": "~",
        "sim": "~", "propto": "~", "equiv": "=="
    }.get(m.group(1), m.group(1)), tex)
    tex = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega|pi|rho|tau|nu|eta|kappa|xi|zeta|chi)\b", lambda m: m.group(1), tex)
    tex = re.sub(r"\\(Alpha|Beta|Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega|Pi)\b", lambda m: m.group(1), tex)
    tex = re.sub(r"\\(infty|infinity)\b", "inf", tex)
    tex = re.sub(r"\\(mathbb)\{([^}]*)\}", r"\2", tex)
    tex = re.sub(r"\\(frac)\{([^}]*)\}\{([^}]*)\}", r"(\2/\3)", tex)
    tex = re.sub(r"\\(sqrt)\{([^}]*)\}", r"sqrt(\2)", tex)
    tex = re.sub(r"\\(sum|prod|int|max|min|arg\s*min|arg\s*max|log|exp|sup|lim)\b", lambda m: m.group(1), tex)
    tex = re.sub(r"\\(reals)\b", "R", tex)
    tex = re.sub(r"\\(ell)\b", "l", tex)

    # Remove \newline, \\
    tex = re.sub(r"\\(newline)\b", "\n", tex)
    tex = re.sub(r"\\\\", "\n", tex)

    # Remove remaining unknown backslash commands (conservative: only single-word commands)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})*", "", tex)

    # Clean up braces, $ signs
    tex = tex.replace("{", "").replace("}", "")
    tex = re.sub(r"\$+", "", tex)

    # Clean up tildes used as non-breaking spaces
    tex = tex.replace("~", " ")

    # 11. Remove blank lines that appear more than twice consecutively
    tex = re.sub(r"\n{4,}", "\n\n\n", tex)

    # Clean up whitespace on lines
    lines = tex.split("\n")
    lines = [line.rstrip() for line in lines]
    tex = "\n".join(lines)

    # Final cleanup of excessive blank lines
    tex = re.sub(r"\n{4,}", "\n\n\n", tex)

    return tex.strip() + "\n"


def main():
    tex = INPUT.read_text(encoding="utf-8")
    txt = convert(tex)
    OUTPUT.write_text(txt, encoding="utf-8")
    print(f"Written {len(txt)} chars ({len(txt.splitlines())} lines) to {OUTPUT}")


if __name__ == "__main__":
    main()
