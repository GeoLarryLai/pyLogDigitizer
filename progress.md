
## Synthetic-scan pilot (B-first), 2026-11-04
- synth_scan v2: `render_scan_tile` fitted to real 1-bit scan stats (`scan_stats`, ELY_1/Hall_D_4/Swanson_1/WEATHERFORD). Realism gate: pass, with caveats (real curves have more texture, synthetic speckle and holes run higher).
- `suspect_reasons` filter (stairs/linear/sawtooth): flags ILD/SP refs that are piecewise linear (USA_A_16, GOHEEN, Hall_D_4 SP, BACASTOW, MAKEMSON, Swanson, George_B_Strong) and sawtooth refs (ELY_1 RES, part of Chitwood DT). Kept 84 segments from 46 curves, no WEATHERFORD.
- Local-only torch training (pld_bench/synth_pilot/train.py): 387k-param UNet, CE + soft clDice, 3000 tiles, 3000 steps, 38 min on CPU. ONNX 1.5 MB.
- Synthetic val IoU: paper .983, grid .944, curveA .584, curveB .315, text .712, curve_any .818.
- New `scan_segmenter` (numpy + onnxruntime): tiled inference, ~0.06 s per 512² crop, 8.4 s for the full WEATHERFORD panel.
- WEATHERFORD GR with idea C on the model curve mask: full r_reg 0.424 (old 0.458). Crop 3150-3220 r_reg 0.879 (old 0.865). No-wrap 3150-3200 r 0.955 (old 0.928, panel C 0.938).
