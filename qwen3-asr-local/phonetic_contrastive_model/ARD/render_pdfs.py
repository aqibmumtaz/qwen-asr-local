#!/usr/bin/env python3
"""Render phonetic contrastive model docs to PDF via WeasyPrint."""
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

ARD = Path(__file__).resolve().parent

DOCS = [
    ("phonetic-contrastive-model-architecture.md", "phonetic-contrastive-model-architecture.pdf",
     "Phonetic Contrastive Model — Architecture"),
    ("phonetic-contrastive-model-technical-doc.md", "phonetic-contrastive-model-technical-doc.pdf",
     "Phonetic Contrastive Model — Technical Document"),
]

CSS_STR = """
@page {
    size: A4;
    margin: 1.6cm 1.8cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
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
    background: #f4f4f4;
    padding: 0.15em 0.3em;
    border-radius: 3px;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 0.8em;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-wrap: break-word;
}

pre code {
    background: none;
    padding: 0;
    font-size: 8.5pt;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #ccc;
    padding: 0.4em 0.6em;
    text-align: left;
}

th {
    background: #f0f0f0;
    font-weight: bold;
}

tr:nth-child(even) { background: #fafafa; }

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
}

strong { color: #111; }

blockquote {
    border-left: 3px solid #4EC5F1;
    margin: 0.8em 0;
    padding: 0.4em 1em;
    background: #f8fcfe;
    color: #555;
}
"""

css = CSS(string=CSS_STR)

for md_name, pdf_name, title in DOCS:
    md_path = ARD / md_name
    pdf_path = ARD / pdf_name

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body>{html_body}</body></html>"""

    HTML(string=full_html).write_pdf(str(pdf_path), stylesheets=[css])
    print(f"Saved: {pdf_path}")
