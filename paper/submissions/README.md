# Workshop submissions (CoRL 2026)

Both folders are self-contained (compile `main.tex`; the built PDF is `paper.pdf`; the matching `.zip` uploads to
Overleaf). Both are copies of `../main_phase1.tex` (the version with the Phase-1 results); they differ only in one
sentence of the introduction that matches each workshop's theme.

| folder | workshop | deadline | intro framing |
|---|---|---|---|
| `physwm_oct7/` | Bringing Physics Simulation and World Models Together for Robotic Manipulation (corl26ws-physwm.github.io) | Oct 7, 23:59 AoE (Oct 8, 07:59 EDT) | "the physical fidelity of world models" |
| `robots_need_wm_oct12/` | Do Robots Need World Models? (do-robots-need-world-models.github.io) | Oct 12 | "the debate on whether robots need world models" |

The Oct-12 version may still be updated (e.g. the WM readout-fix result if panel 7 confirms it); the Oct-7 version
is final once submitted. Both workshops are non-archival; the physwm workshop explicitly welcomes papers under
review elsewhere. Both are on Nov 12, 2026 in Austin.

## Camera-ready (accepted, website tier): `robots_need_wm_camera_ready/`

Due 2026-11-06 23:59 PT on OpenReview. Copy of `robots_need_wm_oct12/`, de-anonymised (`[preprint]` style option, workshop
footer), rewritten for readability and the two reviews (uQ9S20, TT3c23): abstract/Table 2 no longer mix evaluation sets
(fresh-set numbers from results/jenga/phase1e_panel{6,7}_result.json), rare-branch oversampling result added, limitations
and AI usage statement outside the 4 pages, Appendix A answers the reviewer's open questions, Appendix B is the set-C
operating-point sweep (eval/cr_setc_sweep.py -> results/jenga/cr_setc_sweep.json). Fig. 1(b) regenerated with larger type
(make_fig_futures.py, CPU). Poster: `poster/poster.html` -> `poster/poster.pdf` (A0, headless Chrome print).
