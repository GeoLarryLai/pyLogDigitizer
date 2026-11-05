
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
