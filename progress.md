
## Synthetic-scan pilot (B-first), 2026-11-04
- synth_scan v2: `render_scan_tile` fitted to real 1-bit scan stats (`scan_stats`, ELY_1/Hall_D_4/Swanson_1/WEATHERFORD). Realism gate: pass, with caveats (real curves have more texture, synthetic speckle and holes run higher).
- `suspect_reasons` filter (stairs/linear/sawtooth): flags ILD/SP refs that are piecewise linear (USA_A_16, GOHEEN, Hall_D_4 SP, BACASTOW, MAKEMSON, Swanson, George_B_Strong) and sawtooth refs (ELY_1 RES, part of Chitwood DT). Kept 84 segments from 46 curves, no WEATHERFORD.
- Local-only torch training (pld_bench/synth_pilot/train.py): 387k-param UNet, CE + soft clDice, 3000 tiles, 3000 steps, 38 min on CPU. ONNX 1.5 MB.
- Synthetic val IoU: paper .983, grid .944, curveA .584, curveB .315, text .712, curve_any .818.
- New `scan_segmenter` (numpy + onnxruntime): tiled inference, ~0.06 s per 512² crop, 8.4 s for the full WEATHERFORD panel.
- WEATHERFORD GR with idea C on the model curve mask: full r_reg 0.424 (old 0.458). Crop 3150-3220 r_reg 0.879 (old 0.865). No-wrap 3150-3200 r 0.955 (old 0.928, panel C 0.938).

## Synthetic pilot 2 (2026-11-04)
- synth_scan: textured strokes (correlated width wobble, edge noise, ink blobs), despeckle and then real-range speckle/holes (10% of tiles heavy-speckle), solid grid lines. Synthetic speckle 53 and holes 9.5 per Mpx (real typical 27-145 and 9.5-44).
- Segmenter v2 with 4 classes (paper/grid/curve/text), same UNet, 3000 steps in 29.5 min on box CPU, ONNX 1.5 MB. Val IoU: paper .984, grid .965, curve .775, text .730.
- `apply_identity` + `appearance_trace="model"`: joint candidate re-trace on curve ink, each later candidate excluded from claimed strokes except near crossings.
- WEATHERFORD r_reg: GR 0.400 (old 0.458, pilot 1 0.424); NPHI 0.491 (old 0.485); DPHI 0.703 (old 0.748). GR crop 3150-3220 r_reg 0.878; no-wrap 3150-3200 r 0.936 (old 0.928, pilot 1 0.955).
- Still failing: in the NPHI/DPHI track both candidates ride the same (left) curve for most of the depth, and the right curve is never traced, so the identity step cannot help. The GR backup wrap near 3205 ft is still flattened at ~140 API.

## Seeded multi-curve + explicit wrap (2026-11-05)
- `find_seeds` / `seeded_multi_trace` (opt-in `seed_curves=True` with `appearance_trace="model"`): independent seeds from distinct curve-ink runs; widest curve traced first; later curves are excluded from claimed strokes except near crossings. Roles (which curve is the job's pick) are matched to the pick-rule candidates by overlap, because the WEATHERFORD track2 header has no detectable legend (`legend_styles` returns []).
- `wrap_backup`: explicit wrap transition, valued on the backup scale.
- WEATHERFORD r_reg: NPHI 0.598 (old 0.485, pilot 2 0.491); DPHI 0.658 (old 0.748, pilot 2 0.703); GR 0.400 with seeding off (pilot 2 0.400, old 0.458); GR with seeding on 0.251. GR crop r_reg 0.884, no-wrap r 0.936.
- Still failing: the wrap near 3207 ft is detected but valued ~152, not ~300, because the backup trace is limited to the left half and the real backup curve sweeps to the right edge. In the NPHI/DPHI track, candidates still share one curve for long stretches. Shape sanity check only: candidate 0 matches DPHI's shape better (|r| 0.83) than candidate 1 does (0.67), so the DPHI role is probably on the wrong trace. No re-assignment was made from the refs.

## Seed/wrap round 2 (2026-11-05)
- wrap_backup: the backup trace now runs across the full track width with the main stroke's own edge run masked out. A false-wrap check (backup ink within ±2 px for ≥60% of samples) and a minimum span of 3 samples were added.
  - WEATHERFORD GR: the 1-sample wrap at 3630 ft is dropped (KGS shows 141 there, so no wrap).
  - The 4639 ft wrap is kept, valued 150-157 against KGS 144-159.
  - 3207.5 ft is still valued about 152. In the ink crop the main tracer runs to the right edge, but no backup stroke enters from the left. The KGS peak of 297 looks like a depth offset or tracer excursion, not a visible wrap.
  - GR r_reg 0.400, crop 0.884 (unchanged).
- Seeding changes:
  - Seeds closer to the track edge than 2% of its width are rejected.
  - Ink claimed by an earlier trace is allowed only in crossing windows of 15 samples or fewer.
  - A trace is re-seeded where it collapses within 6 px or loses ink for 40 samples or more.
  - New option seed_role="dashed|solid|thick|thin" assigns identity by line style; values are never used.
- Clipping: model mode writes null where the trace runs along a track edge outside wrap spans. NPHI has 110 null samples and DPHI 21.
- LAS resample: step is 0.5 ft by default, with 0.25 ft as the option (anything else raises an error). Interpolation is linear between neighbouring trace samples only, and a gap or clipped neighbour gives a null.
- r_reg: NPHI 0.475 (seed_wrap 0.598), DPHI 0.820 (0.658).
  - On WEATHERFORD the style evidence is weak: gap share is 0.022 vs 0.052 and widths are 13 vs 10 px. The thin trace (cand 1) fits both KGS curves better (|r| NPHI 0.55, DPHI 0.83), so "dashed=NPHI" gives the wrong NPHI.
  - The traces still collapse within 6 px for 12% of the depth.
- Figures: /workspace/pld_viz_out/seed_wrap2/.

## Curve-identity segmenter (2026-11-05)
- 5-class U-Net (paper/grid/curveA/curveB/text) trained on 3000 synthetic tiles from KGS curves, with WEATHERFORD excluded. curveA is the heavier, mostly solid stroke; curveB is the stroke 0.7x as wide, solid, dashed or dotted. Training ran 3000 steps on the box CPU in 2229 s.
  - Synthetic val IoU: paper .985, grid .966, curveA .573, curveB .277, text .722, any-curve .788. A/B identity accuracy on curve pixels is 0.73.
  - The model is in /workspace/pld_bench/curve_id_seg/seg_unet.onnx.
- Code: seeded_multi_trace(class_inks=...) seeds and traces each class on its own ink, with mutual exclusion. digitize_curve(class_role="A"|"B") turns this on, and scan_segmenter.curve_class_probability returns the per-class maps.
- WEATHERFORD gate (figures first): 90% of track-2 curve ink is labelled A. Pixels under the NPHI/DPHI traces are 71%/86% A and only 4% B, so the model does not separate the two curves on the real scan.
  - No NPHI/DPHI benchmark was run, because class-seeded tracing on 4% B ink would trace noise.
  - Figures: /workspace/pld_viz_out/curve_id_seg/seg_classes_{2528,2624}.png.

## Per-well line hint (2026-11-05)
- New option digitize_curve(line_hint=thin|thick|dashed|solid|left|right) for scans without a legend. It only says which seeded line this curve is and never uses LAS values. The curve-id model stays opt-in and is unused by default, and style tuning is stopped.
- WEATHERFORD: the hint NPHI=thin, DPHI=thick picks exactly the same seeded lines as the seed_wrap2 style rule. The thin line (9.6 px) is also the dashed one, and the thick line (13.1 px) the solid one. Scores are therefore identical: NPHI 0.475, DPHI 0.820 (old 0.485/0.748). I did not rerun them; the figures in /workspace/pld_viz_out/hint_nphi/ are made from the seed_wrap2 traces.

## Easier KGS ink scans, default model path (2026-11-05)
- Settings as on WEATHERFORD (model mode, curve wraps and nulls at clipped edges), with no seeding and no tuning.
- Chitwood_12 GR:2: r_reg 0.993, 53 s run, 329 clipped samples set to null.
- Swanson_1 GR: r_reg 0.736, 9 s run. The scan covers only about 500-1050 ft. Over that stretch the trace follows the ink closely. It skips across with straight connector lines where the curve leaves the track, which may be a backup that was missed.
- Figures: /workspace/pld_viz_out/easy_ink/.

## Swanson spikes + easy ink 2 (2026-11-05)
- New spike_excursions step for model mode with a backup scale. Some off-scale spikes are drawn as near-horizontal strokes, too fast for the row-wise tracer, which bridges them with straight connectors. Where a curve-ink run starts at the trace and reaches the wrap edge, the value is read from a backup run entering from the opposite edge; if there is none, the sample is null.
- Swanson_1 GR: r_reg 0.736 -> 0.736. Only about 800 ft fires, and it is nulled (KGS 188 there).
  - At 643 ft the spike strokes in the 1-bit scan merge with horizontal grid lines and are labelled grid, so the curve ink loses them.
  - Running on all ink instead gave false hits on grid lines.
  - Most connectors are therefore not fixed.
- Easy GR scans, default model path, no tuning (r_reg): ELY_1 0.530, GOHEEN_UNIT_1 0.617, Hall_D_4 0.630, USA_A_16 0.735, Minzer_1 0.080.
  - Minzer: the trace misses the backup curve near the left edge and bridges spikes. Around 3200-3800 ft it reads about 15 where KGS reads about 120 (to check: section/scale or the backup).
  - Earlier: Chitwood_12 0.993.
- Figures: /workspace/pld_viz_out/easy_ink2/.

## Overflow wraps across connectors (2026-11-05)
- New sw.ink_jump_wraps: wrap detection uses only the inked trace samples. A row-wise tracer draws a slanted connector across empty paper instead of jumping, which hid the wrap. Connector samples that cross more than 30% of the width are written as null.
- r_reg before -> after: Chitwood 0.993 -> 0.993, ELY 0.530 -> 0.508, GOHEEN 0.617 -> 0.617, Hall 0.630 -> 0.630, Minzer 0.080 -> 0.200, Swanson 0.736 -> 0.736, USA_A_16 0.735 -> 0.737.
- Minzer root cause: from about 3200 to 3800 ft the main curve is off-scale on the right and only the backup stroke near the left edge is inked. The wrap entry is not detected there, so the backup is read on the main scale (15-30 where KGS reads 120-140).
- Not done: segmenter retrain with synthetic overflow spikes.
- Figures: /workspace/pld_viz_out/easy_ink3/.

## Overflow segmenter + continuity wraps (2026-11-05)
- synth_scan.render_scan_tile(overflow=p): adds near-flat fast spikes running to a track edge, backup strokes re-entering from the opposite edge, and long backup-only stretches near an edge. All are labelled curve and cross the grid.
  - 4-class model retrained on Larry's Mac (MPS, 3000 steps, 284 s); weights copied back and exported to ONNX on the box.
  - Model: /workspace/pld_bench/overflow_seg/seg_unet.onnx. Synthetic val IoU: paper .980, grid .957, curve .770, text .729.
- ink_jump_wraps additions:
  - Border-hugging samples and isolated specks are ignored.
  - Continuity rule: the curve reaches the wrap edge zone, vanishes for at least 10 samples, and comes back near the opposite edge. This counts as an exit even when no exit was drawn.
  - A cancelled wrap is kept up to its last entry-edge sample instead of being dropped.
  - A wrap is accepted only if the backup stays near the entry edge after landing; this removed a false Hall wrap.
- USA_A_16 runtime: 487 s -> 134 s (vectorised on-ink check; spike scan prefiltered to bands touching the edge).
- r_reg, easy_ink3 -> easy_ink4 (new model plus rules): Chitwood 0.993 -> 0.989, Swanson 0.736 -> 0.803, USA_A_16 0.737 -> 0.797, Hall 0.630 -> 0.630, GOHEEN 0.617 -> 0.595, ELY 0.508 -> 0.421, Minzer 0.200 -> 0.494.
- Minzer:
  - Over 3250-3780 ft the backup values match KGS within about 5 (for example 116 vs 116, 118 vs 118). The CPS label on the print uses the same numbers as the LAS API curve, so no unit conversion is applied; we have not confirmed whether CPS here is just a label.
  - The 300-700 ft overflow is real: KGS median there is 115, above the 110 main-scale maximum.
- ELY: a false wrap opens at 4458 ft to the end of the log. A low GR curve near the left edge looks like a backup, so the rule cannot tell them apart. Part of the ELY drop probably comes from the new model; it was not isolated.
- Figures: /workspace/pld_viz_out/easy_ink4/.

## Cleanup, bundled segmenter, overlay QA, manual node editor (2026-11-05)
- Removed abandoned paths: 5-class curve-id segmenter (curve_class_probability, class_role/seed_role), style/appearance tuning (apply_appearance, class_inks), grid-residual modes and the `grid_layer` module + their tests. `appearance_trace` is now "model" or off.
- Bundled models (1.5 MB each) in `src/pylogdigitizer/models/`: `scan_seg_v2.onnx` (default) and `scan_seg_overflow.onnx` (opt-in, `default_model("overflow")`).
- Default kept on the old model because it holds Chitwood at 0.993; overflow model is better on average:

| scan | default (v2) | overflow |
|---|---|---|
| Chitwood_12 | 0.993 | 0.989 |
| Swanson_1 | 0.736 | 0.803 |
| USA_A_16 | 0.793 | 0.797 |
| Hall_D_4 | 0.630 | 0.630 |
| GOHEEN_UNIT_1 | 0.617 | 0.595 |
| ELY_1 | 0.367 | 0.421 |
| Minzer_1 | 0.492 | 0.494 |
| mean | 0.661 | 0.676 |

- `scripts/train_scan_segmenter.py` + `notebooks/03_train_scan_segmenter.ipynb`: write to a user dir; refuse to overwrite the bundled model unless `overwrite_default=True`.
- `qc_overlay.overlay_scan` (raster) and `pdf_vector.overlay.overlay_pdf_page` (dev/0.2-pdf): track edges, grid lines and traces over the scan / page vectors.
- Manual node editor (`AssistedSession.node_snap`, `POST /fix/node`, UI tool D): drop/drag a node, local ink snap inside the circle, reconnects above/below, ~2 ms, undo/session round trip. Demo: `scripts/demo_node_editor.py`.
- Suite: 265 passed, 2 skipped, 1 xfailed.
