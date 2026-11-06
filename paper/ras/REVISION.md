# RAS revision — 6 November 2026

The abstract, contributions, gradient ablation discussion, practical comparison,
limitations, conclusion and highlights now reflect the available evidence.
Section 7.3.1, Table 4 and Figure 4 report the new controller-cost sweep.
The look-alike gradient ablation is an additional analysis of existing runs.

The main claim is trusted-reference, whole-log curation under a consistent
majority fault. Gradients improve completion in the look-alike experiment,
but are not necessary for the thrust-gain curation result. FACT and look-ahead
reweighting exhibit a safety–completion tradeoff at all three tested costs;
no equal-risk superiority or Pareto dominance is established.

Validation: original numerical checks and the new generated table/interval
checks pass. Tectonic compiles the manuscript with no unresolved references
or citations. All 40 pages were rendered and visually inspected, including
the new table and figure. Small existing TeX box warnings do not cause
visible clipping. The PDF is `ras_manuscript_revised.pdf`.

The source includes the two generated TeX inputs and new figure. The existing
submission assembler now inlines those inputs and copies the figure; its
pdflatex-based full submission-bundle build was not run in this environment.
The revised PDF was built directly with Tectonic.
