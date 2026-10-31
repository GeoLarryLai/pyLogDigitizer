// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! curve_tracing_rs: Rust core for pylogdigitizer curve tracing (DP path through a cost image).
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use pyo3::prelude::*;

/// Dynamic-programming trace: one column per row minimizing cost + smooth*|dx| (|dx| <= max_step).
/// Mirrors pylogdigitizer.curve_tracing.trace_dp_python.
#[pyfunction]
fn trace_dp<'py>(py: Python<'py>, cost: PyReadonlyArray2<'py, f64>, max_step: usize, smooth: f64)
    -> PyResult<Bound<'py, PyArray1<i32>>> {
    let c = cost.as_array();
    let (h, w) = c.dim();
    let mut acc: Vec<f64> = c.row(0).to_vec();
    let mut back = vec![0i32; h * w];
    for y in 1..h {
        let mut next = vec![f64::INFINITY; w];
        for x in 0..w {
            let lo = x.saturating_sub(max_step);
            let hi = (x + max_step).min(w - 1);
            for px in lo..=hi {
                let v = acc[px] + smooth * (x as f64 - px as f64).abs();
                if v < next[x] { next[x] = v; back[y * w + x] = px as i32; }
            }
            next[x] += c[[y, x]];
        }
        acc = next;
    }
    let mut path = vec![0i32; h];
    let mut x = acc.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
    for y in (0..h).rev() { path[y] = x as i32; if y > 0 { x = back[y * w + x] as usize; } }
    Ok(PyArray1::from_vec_bound(py, path))
}

/// DP trace with segment cost: a jump from px to x between rows also pays
/// seg_weight * mean(min(cost[y], cost[y-1]) over columns px..=x), so horizontal jumps must run
/// along ink. Same algorithm as pylogdigitizer.curve_tracing._dp_numba.
#[pyfunction]
fn trace_dp_seg<'py>(py: Python<'py>, cost: PyReadonlyArray2<'py, f64>, max_step: usize, smooth: f64,
                     seg_weight: f64) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let c = cost.as_array();
    let (h, w) = c.dim();
    let path = py.allow_threads(|| {
        let mut acc: Vec<f64> = c.row(0).to_vec();
        let mut nxt = vec![0f64; w];
        let mut pref = vec![0f64; w + 1];
        let mut back = vec![0i32; h * w];
        for y in 1..h {
            for x in 0..w { pref[x + 1] = pref[x] + c[[y, x]].min(c[[y - 1, x]]); }
            for x in 0..w {
                let lo = x.saturating_sub(max_step);
                let hi = (x + max_step).min(w - 1);
                let mut best = f64::INFINITY; let mut arg = x;
                for px in lo..=hi {
                    let d = if x > px { x - px } else { px - x };
                    let mut v = acc[px] + smooth * d as f64;
                    if d > 0 {
                        let (a, b) = if px < x { (px, x) } else { (x, px) };
                        v += seg_weight * (pref[b + 1] - pref[a]) / ((b - a + 1) as f64);
                    }
                    if v < best { best = v; arg = px; }
                }
                nxt[x] = best + c[[y, x]]; back[y * w + x] = arg as i32;
            }
            std::mem::swap(&mut acc, &mut nxt);
        }
        let mut path = vec![0i32; h];
        let mut bx = 0usize;
        for x in 0..w { if acc[x] < acc[bx] { bx = x; } }
        path[h - 1] = bx as i32;
        for y in (1..h).rev() { path[y - 1] = back[y * w + path[y] as usize]; }
        path
    });
    Ok(PyArray1::from_vec_bound(py, path))
}

/// DP trace with segment cost and curve-identity term: entering a cell whose stroke id
/// (labels, 0 = no ink) differs from the previous cell's id costs switch_cost.
/// Same algorithm as pylogdigitizer.curve_tracing._dp_numba.
#[pyfunction]
fn trace_dp_lab<'py>(py: Python<'py>, cost: PyReadonlyArray2<'py, f64>, labels: PyReadonlyArray2<'py, i32>,
                     max_step: usize, smooth: f64, seg_weight: f64, switch_cost: f64) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let c = cost.as_array();
    let l = labels.as_array();
    let (h, w) = c.dim();
    let c_vec: Vec<f64> = if let Some(s) = c.as_slice() { s.to_vec() } else { c.iter().copied().collect() };
    let l_vec: Vec<i32> = if let Some(s) = l.as_slice() { s.to_vec() } else { l.iter().copied().collect() };
    let path = py.allow_threads(move || {
        let mut acc: Vec<f64> = c_vec[0..w].to_vec();
        let mut nxt = vec![0f64; w];
        let mut pref = vec![0f64; w + 1];
        let mut back = vec![0i32; h * w];
        let inv_seg: Vec<f64> = (0..=w).map(|d| seg_weight / ((d + 1) as f64)).collect();
        let sm_tab: Vec<f64> = (0..=w).map(|d| smooth * (d as f64)).collect();
        for y in 1..h {
            let cy = &c_vec[y * w .. (y + 1) * w];
            let cy1 = &c_vec[(y - 1) * w .. y * w];
            let ly = &l_vec[y * w .. (y + 1) * w];
            let ly1 = &l_vec[(y - 1) * w .. y * w];
            let back_row = &mut back[y * w .. (y + 1) * w];
            for x in 0..w { pref[x + 1] = pref[x] + cy[x].min(cy1[x]); }
            for x in 0..w {
                let lo = x.saturating_sub(max_step);
                let hi = (x + max_step).min(w - 1);
                let lx = ly[x];
                let use_sw = switch_cost != 0.0 && lx > 0;
                let mut best = f64::INFINITY; let mut arg = x;
                let pref_x1 = pref[x + 1];
                let pref_x = pref[x];
                for px in lo..x {
                    let d = x - px;
                    let mut v = acc[px] + sm_tab[d] + (pref_x1 - pref[px]) * inv_seg[d];
                    if use_sw && lx != ly1[px] { v += switch_cost; }
                    if v < best { best = v; arg = px; }
                }
                {
                    let mut v = acc[x];
                    if use_sw && lx != ly1[x] { v += switch_cost; }
                    if v < best { best = v; arg = x; }
                }
                for px in (x + 1)..=hi {
                    let d = px - x;
                    let mut v = acc[px] + sm_tab[d] + (pref[px + 1] - pref_x) * inv_seg[d];
                    if use_sw && lx != ly1[px] { v += switch_cost; }
                    if v < best { best = v; arg = px; }
                }
                nxt[x] = best + cy[x]; back_row[x] = arg as i32;
            }
            std::mem::swap(&mut acc, &mut nxt);
        }
        let mut path = vec![0i32; h];
        let mut bx = 0usize;
        for x in 0..w { if acc[x] < acc[bx] { bx = x; } }
        path[h - 1] = bx as i32;
        for y in (1..h).rev() { path[y - 1] = back[y * w + path[y] as usize]; }
        path
    });
    Ok(PyArray1::from_vec_bound(py, path))
}


fn deg_tan(a: f64) -> f64 { (a * (std::f64::consts::PI / 180.0)).tan() }

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Piecewise (per-block) deskew of a bilevel image (u8 0/1). Mirrors
/// pylogdigitizer.scan_cleanup._deskew_blocks_nb: per block, the shear angle maximising the sum of
/// squares of the horizontal projection profile (columns c0..c1); angles median-smoothed over
/// +-smooth blocks; each block sheared about the sample centre. Returns (deskewed u8, angles).
#[pyfunction]
fn deskew_blocks<'py>(py: Python<'py>, ink: PyReadonlyArray2<'py, u8>, block: usize, c0: usize, c1: usize,
                      angles: PyReadonlyArray1<'py, f64>, smooth: usize)
    -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f64>>)> {
    let a = ink.as_array();
    let (h, w) = a.dim();
    let img: Vec<u8> = if let Some(s) = a.as_slice() { s.to_vec() } else { a.iter().copied().collect() };
    let angs: Vec<f64> = angles.as_array().iter().copied().collect();
    let (out, ang) = py.allow_threads(move || {
        let block = block.max(1);
        let n_blocks = (h + block - 1) / block;
        let max_abs = angs.iter().fold(0f64, |m, v| m.max(v.abs()));
        let max_s = (w as f64 * deg_tan(max_abs)).ceil() as i64 + 2;
        let cw = c1.saturating_sub(c0);
        let raw: Vec<f64> = (0..n_blocks).into_par_iter().map(|b| {
            let s = b * block; let e = (s + block).min(h); let bh = e - s;
            let mut ys: Vec<i64> = Vec::new(); let mut xs: Vec<usize> = Vec::new();
            for y in 0..bh { let row = &img[(s + y) * w..(s + y + 1) * w];
                for x in c0..c1 { if row[x] != 0 { ys.push(y as i64); xs.push(x - c0); } } }
            if ys.is_empty() { return 0.0; }
            let mut prof = vec![0i64; bh + 2 * max_s as usize + 2];
            let mut shifts = vec![0i64; cw];
            let mut best = -1.0f64; let mut best_a = 0.0f64;
            for &an in angs.iter() {
                let t = deg_tan(an);
                for x in 0..cw { shifts[x] = (x as f64 * t).round_ties_even() as i64 + max_s; }
                prof.iter_mut().for_each(|p| *p = 0);
                for i in 0..ys.len() { prof[(ys[i] + shifts[xs[i]]) as usize] += 1; }
                let v: f64 = prof.iter().map(|&p| { let f = p as f64; f * f }).sum();
                if v > best { best = v; best_a = an; }
            }
            best_a
        }).collect();
        let ang: Vec<f64> = if smooth > 0 && n_blocks > 2 {
            (0..n_blocks).map(|i| { let lo = i.saturating_sub(smooth); let hi = (i + smooth + 1).min(n_blocks);
                let mut v = raw[lo..hi].to_vec(); median(&mut v) }).collect()
        } else { raw.clone() };
        let xc = (c0 + c1) as f64 / 2.0;
        let mut out = vec![0u8; h * w];
        out.par_chunks_mut(block * w).enumerate().for_each(|(b, chunk)| {
            let s = b * block; let t = deg_tan(ang[b]);
            let sh: Vec<i64> = (0..w).map(|x| -(((x as f64 - xc) * t).round_ties_even() as i64)).collect();
            let bh = chunk.len() / w.max(1);
            for y in 0..bh { let dy = (s + y) as i64;
                for x in 0..w { let sy = dy + sh[x];
                    if sy >= 0 && (sy as usize) < h { chunk[y * w + x] = img[sy as usize * w + x]; } } }
        });
        (out, ang)
    });
    let arr = PyArray1::from_vec_bound(py, out).reshape([h, w])?;
    Ok((arr, PyArray1::from_vec_bound(py, ang)))
}

#[pymodule]
fn curve_tracing_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trace_dp, m)?)?;
    m.add_function(wrap_pyfunction!(trace_dp_seg, m)?)?;
    m.add_function(wrap_pyfunction!(trace_dp_lab, m)?)?;
    m.add_function(wrap_pyfunction!(deskew_blocks, m)?)?;
    Ok(())
}
