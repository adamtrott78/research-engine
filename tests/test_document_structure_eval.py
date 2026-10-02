from research_engine.document_structure_eval import (
    extract_mathpix_equations,
    extract_mathpix_figure_captions,
    extract_mathpix_headings,
    extract_mathpix_tables,
)


def test_mathpix_reference_structure_extractors_keep_content_units_separate():
    markdown = r"""
# Overview

\section{Methods}

\[
E = mc^2
\]

\begin{table}
\caption{Results}
\begin{tabular}{cc}
Model & Score \\
A & 0.9
\end{tabular}
\end{table}

\begin{figure}
\includegraphics{figure.png}
\caption{System architecture and signal flow.}
\end{figure}
"""

    assert extract_mathpix_headings(markdown) == ["Overview", "Methods"]
    assert extract_mathpix_equations(markdown) == ["E = mc^2"]
    assert extract_mathpix_tables(markdown) == [
        [["Model", "Score"], ["A", "0.9"]]
    ]
    assert extract_mathpix_figure_captions(markdown) == [
        "System architecture and signal flow."
    ]
