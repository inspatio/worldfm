"""
run_vbench.py — WorldFM VBench i2v evaluation.

Loads external repos and the WorldFM inference service once, then runs the
full pipeline (panogen → moge → render → infer) for every VBench crop image.

Usage:
    python run_vbench.py vbench [VBENCH_INFO_JSON] [--output_dir ...] [--image_types ...]

    python run_vbench.py run --meta demo/meta.json          # single-image run
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import fire
import numpy as np
import torch

import run_pipeline as _p

_WORLDFM_ROOT      = _p.WORLDFM_ROOT
_VBENCH_ROOT       = _WORLDFM_ROOT.parent / 'VBench' / 'vbench2_beta_i2v' / 'vbench2_beta_i2v' / 'data'
_DEFAULT_INFO_JSON = str(_VBENCH_ROOT / 'i2v-bench-info.json')
_DEFAULT_CROP_DIR  = str(_VBENCH_ROOT / 'crop')


# ─────────────────────────────────────────────────────────────────────────────

def vbench_batch(
    vbench_info_json=None,
    output_dir='results_vbench/videos',
    image_types='indoor,scenery',
    resolution='1-1',
    crop_dir=None,
    fps=30,
    step=None,
    cfg_scale=None,
):
    info_json  = os.path.abspath(vbench_info_json or _DEFAULT_INFO_JSON)
    crop_base  = os.path.abspath(crop_dir or _DEFAULT_CROP_DIR)
    image_dir  = os.path.join(crop_base, resolution)
    out_dir    = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path    = os.path.join(os.path.dirname(out_dir), 'vbench_stats.csv')
    stats_is_new  = not os.path.exists(stats_path)
    stats_f       = open(stats_path, 'a', newline='', encoding='utf-8')
    stats_w       = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(['task_idx', 'file', 'type', 'duration_s', 'vram_gb', 'ram_gb', 'out_path', 'status'])

    if not os.path.isfile(info_json):
        sys.exit(f'[vbench] ERROR: info JSON not found: {info_json}')
    if not os.path.isdir(image_dir):
        sys.exit(f'[vbench] ERROR: crop dir not found: {image_dir}')

    # load demo K / c2w trajectory
    with open(_WORLDFM_ROOT / 'demo' / 'meta.json', encoding='utf-8') as f:
        _demo = json.load(f)
    K        = np.asarray(_demo['K'],   dtype=np.float64)
    c2w_list = [np.asarray(c, dtype=np.float64) for c in _demo['c2w']]

    # parse allowed image types
    allowed = {t.strip() for t in image_types.split(',') if t.strip()} if image_types else None

    with open(info_json, encoding='utf-8') as f:
        entries = json.load(f)

    seen, images = set(), []
    for e in entries:
        fn = e.get('file_name', '')
        if fn in seen:
            continue
        if allowed and e.get('type') not in allowed:
            continue
        seen.add(fn)
        images.append({'file': fn, 'type': e.get('type', '')})

    total = len(images)
    print(f'\n[vbench] {total} images  (types: {image_types}  resolution: {resolution})')
    for i, img in enumerate(images):
        ok = '' if os.path.isfile(os.path.join(image_dir, img['file'])) else '  MISSING'
        print(f'  [{i+1:3}/{total}] ({img["type"]:<8}){ok}  {img["file"]}')
    print()

    # ── load external repos + WorldFM model once ──────────────────────────────
    cfg = _p.DEFAULT_CFG
    if step is not None or cfg_scale is not None:
        from omegaconf import OmegaConf
        overrides = {}
        if step      is not None: overrides['step']      = step
        if cfg_scale is not None: overrides['cfg_scale'] = cfg_scale
        cfg = _p.OmegaConf.merge(cfg, _p.OmegaConf.create({'worldfm': overrides}))

    _p.setup_external_repos(
        hw_path=str(cfg.submodules.hw_path),
        moge_path=str(cfg.submodules.moge_path),
    )
    print('[vbench] loading WorldFM inference service...')
    svc, wcfg = _p.step4_init(cfg=cfg)
    print('[vbench] WorldFM ready\n')

    generated = errors = skipped = 0
    t_start = time.time()

    for ti, img in enumerate(images):
        image_path = os.path.join(image_dir, img['file'])
        stem       = Path(img['file']).stem
        out_mp4    = os.path.join(out_dir, f'{stem}.mp4')

        if not os.path.isfile(image_path):
            print(f'[vbench] skip {ti+1}/{total}: image not found — {image_path}')
            stats_w.writerow([ti, img['file'], img['type'], '', '', '', '', 'missing'])
            stats_f.flush()
            continue

        if os.path.exists(out_mp4):
            print(f'[vbench] skip {ti+1}/{total}: already done — {stem}.mp4')
            stats_w.writerow([ti, img['file'], img['type'], '', '', '', out_mp4, 'skipped'])
            stats_f.flush()
            skipped += 1
            continue

        done_so_far = generated + errors + skipped
        pct = round(100 * ti / total) if total else 0
        eta = ''
        if done_so_far > 0:
            elapsed = time.time() - t_start
            rem = int(elapsed / done_so_far * (total - done_so_far))
            eta = f'  ETA {rem//3600:02d}h{(rem%3600)//60:02d}m{rem%60:02d}s'
        print(f'[vbench] [{ti+1}/{total}  {pct}%{eta}]  ({img["type"]})  {img["file"]}')

        t0 = time.time()
        try:
            # step 1: perspective → panorama
            tmp_out = _WORLDFM_ROOT / '_vbench_tmp' / stem
            tmp_out.mkdir(parents=True, exist_ok=True)
            panorama_img = _p.step1_panogen(image_path, tmp_out, cfg=cfg)

            # step 2: panorama → depth / PLY / conditions
            pp_result = _p.step2_moge_pipeline(panorama_img, tmp_out, cfg=cfg)

            # step 3: init renderer + condition DB (per image)
            renderer, cond_db, rcfg, S = _p.step3_init(pp_result, cfg=cfg)

            # step 4: render + infer each pose
            frames = []
            for c2w in c2w_list:
                render_u8, cond_nearest = _p.step3_render_one(
                    renderer, cond_db, pp_result, K, c2w,
                    rcfg=rcfg, render_size=S,
                )
                frame = _p.step4_infer_one(svc, render_u8, cond_nearest, wcfg=wcfg)
                frames.append(frame)

            del renderer, cond_db
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # save video
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
            for fr in frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()

            dur  = round(time.time() - t0, 2)
            vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else ''
            try:
                import psutil
                ram = round(psutil.Process().memory_info().rss / 1024**3, 2)
            except Exception:
                ram = ''

            print(f'  done in {dur}s  VRAM {vram}GB  RAM {ram}GB  -> {stem}.mp4')
            stats_w.writerow([ti, img['file'], img['type'], dur, vram, ram, out_mp4, 'ok'])
            generated += 1

        except Exception as exc:
            print(f'  EXCEPTION: {exc}', file=sys.stderr)
            stats_w.writerow([ti, img['file'], img['type'], '', '', '', '', 'error'])
            errors += 1
        finally:
            import shutil
            shutil.rmtree(_WORLDFM_ROOT / '_vbench_tmp' / stem, ignore_errors=True)

        stats_f.flush()

    stats_f.close()
    elapsed_m = round((time.time() - t_start) / 60, 1)
    print(f'\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_m}m')
    print(f'[vbench] stats → {stats_path}')


# ─────────────────────────────────────────────────────────────────────────────

def run(meta, output_dir=None, config='', save_mode='video', fps=30,
        step=None, cfg_scale=None, gpu_index=0):
    """Single-image run (mirrors run_pipeline.py CLI for convenience)."""
    import argparse, types
    args = types.SimpleNamespace(
        meta=meta,
        output_dir=output_dir or str(_p.DEFAULT_CFG.pipeline.output_dir),
        config=config,
        save_mode=save_mode,
        fps=fps,
        step=step      if step      is not None else int(_p.DEFAULT_CFG.worldfm.step),
        cfg_scale=cfg_scale if cfg_scale is not None else float(_p.DEFAULT_CFG.worldfm.cfg_scale),
        gpu_index=gpu_index,
        hw_path=str(_p.DEFAULT_CFG.submodules.hw_path),
        moge_path=str(_p.DEFAULT_CFG.submodules.moge_path),
        moge_pretrained=str(_p.DEFAULT_CFG.moge.pretrained),
        render_size=int(_p.DEFAULT_CFG.render.render_size),
        model_path=str(_p.DEFAULT_CFG.worldfm.model_path),
        vae_path=str(_p.DEFAULT_CFG.worldfm.vae_path),
        image_size=int(_p.DEFAULT_CFG.worldfm.image_size),
        version=str(_p.DEFAULT_CFG.worldfm.version),
    )
    cfg = _p._load_config(args)
    if cfg.pipeline.gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(cfg.pipeline.gpu_index))
    from run_pipeline import main as _main
    sys.argv = ['run_pipeline.py', '--meta', meta]
    return _main()


if __name__ == '__main__':
    fire.Fire({'vbench': vbench_batch, 'run': run})
