
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
