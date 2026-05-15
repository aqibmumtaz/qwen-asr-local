"""Render hindi-to-roman-urdu-design.md to PDF via WeasyPrint."""
import sys
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

SCRIPT_DIR = Path(__file__).resolve().parent
MD_PATH    = SCRIPT_DIR / "hindi-to-roman-urdu-design.md"
PDF_PATH   = SCRIPT_DIR / "hindi-to-roman-urdu-design.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc"],
)

# Styling — Noto fonts have Devanagari, Nastaliq, Latin. Available on Mac via system.
css = CSS(string="""
@page {
    size: A4;
    margin: 1.6cm 1.8cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
    }
    @top-right {
        content: "Hindi → Roman Urdu — Design";
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: 'Helvetica Neue', 'Helvetica', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
}

/* Multi-script fallback fonts for Devanagari + Nastaliq + Latin code samples */
.devanagari, .nastaliq {
    font-family: 'Noto Sans Devanagari', 'Noto Nastaliq Urdu', 'Geeza Pro', serif;
}

h1 {
    font-size: 22pt;
    color: #1a1a1a;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 0.3em;
    margin-top: 0;
}

h2 {
    font-size: 15pt;
    color: #1a1a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
    margin-top: 1.5em;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    color: #333;
    margin-top: 1.2em;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #444;
    margin-top: 1em;
}

p { margin: 0.5em 0; }

code {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 9pt;
    background: #f4f4f6;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    color: #b3273f;
}

pre {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 8.5pt;
    background: #f8f8fa;
    border: 1px solid #e5e5ea;
    border-radius: 4px;
    padding: 0.7em 1em;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: transparent;
    color: #222;
    padding: 0;
    font-size: inherit;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #d0d0d5;
    padding: 0.4em 0.6em;
    text-align: left;
    vertical-align: top;
}

th {
    background: #ececf0;
    font-weight: 600;
    color: #1a1a1a;
}

tr:nth-child(even) td {
    background: #fafafc;
}

hr {
    border: 0;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
}

blockquote {
    border-left: 3px solid #aaa;
    margin: 0.6em 0;
    padding: 0 1em;
    color: #555;
}

ul, ol { margin: 0.4em 0 0.4em 1.5em; }
li { margin-bottom: 0.2em; }

strong { color: #1a1a1a; }
em { color: #444; }

a { color: #2c5aa0; text-decoration: none; }
""")

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hindi → Roman Urdu — Design</title>
</head>
<body>
{html_body}
</body>
</html>"""

HTML(string=html_doc, base_url=str(SCRIPT_DIR)).write_pdf(
    PDF_PATH, stylesheets=[css]
)
print(f"Wrote {PDF_PATH}")
