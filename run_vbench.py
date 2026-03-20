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
import traceback
from pathlib import Path

import cv2
import fire
import numpy as np
import torch

import run_pipeline as _p


# ── interactive camera helpers ────────────────────────────────────────────────

def _start_key_listener():
    """Start a non-blocking pynput listener; returns (held_set, stop_fn)."""
    from pynput.keyboard import Listener
    held = set()

    def on_press(key):
        try:
            held.add(key.char.lower())
        except AttributeError:
            held.add(key)

    def on_release(key):
        try:
            held.discard(key.char.lower())
        except AttributeError:
            held.discard(key)

    listener = Listener(on_press=on_press, on_release=on_release, daemon=True)
    listener.start()
    return held, listener.stop


def _c2w_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-world matrix (camera at origin) for given yaw/pitch angles.

    yaw   — horizontal rotation (radians, 0 = looking in +Z)
    pitch — vertical tilt      (radians, 0 = horizontal, + = up)
    """
    fwd = np.array([np.sin(yaw) * np.cos(pitch),
                    -np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)])
    world_up = np.array([0., 1., 0.])
    right = np.cross(world_up, fwd)
    norm = np.linalg.norm(right)
    if norm < 1e-6:           # looking straight up/down — keep last right
        right = np.array([1., 0., 0.])
    else:
        right /= norm
    up = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = fwd
    return c2w


def _yaw_pitch_from_c2w(c2w: np.ndarray):
    """Extract yaw and pitch from a c2w rotation matrix."""
    fwd = c2w[:3, 2]
    yaw   = np.arctan2(float(fwd[0]), float(fwd[2]))
    pitch = np.arcsin(np.clip(-float(fwd[1]), -1.0, 1.0))
    return yaw, pitch

_WORLDFM_ROOT      = _p.WORLDFM_ROOT
_VBENCH_ROOT       = _WORLDFM_ROOT.parent / 'VBench' / 'vbench2_beta_i2v' / 'vbench2_beta_i2v' / 'data'
_DEFAULT_INFO_JSON = str(_VBENCH_ROOT / 'i2v-bench-info.json')
_DEFAULT_CROP_DIR  = str(_VBENCH_ROOT / 'crop')


def _make_K(render_size: int, fov_deg: float = 70.0) -> np.ndarray:
    """Intrinsic matrix for a square render of side `render_size` with given horizontal FOV."""
    f = (render_size / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    c = render_size / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float64)


def _make_panorama_trajectory(
    num_frames: int,
    seed_idx: int,
    num_seeds: int,
    sweep_deg: float = 90.0,
    pitch_amp_deg: float = 5.0,
) -> list:
    """Smooth yaw-sweep trajectory with yaw offset distributed evenly across seeds.

    seed_idx 0..num_seeds-1 each start at yaw = seed_idx * 360/num_seeds degrees,
    then sweep `sweep_deg` horizontally over `num_frames` frames with a gentle
    sinusoidal pitch oscillation.
    """
    yaw_offset = seed_idx * 2.0 * np.pi / num_seeds
    yaws   = yaw_offset + np.linspace(0.0, np.radians(sweep_deg), num_frames)
    pitches = np.radians(pitch_amp_deg) * np.sin(np.linspace(0.0, np.pi, num_frames))
    return [_c2w_from_yaw_pitch(float(y), float(p)) for y, p in zip(yaws, pitches)]


# ─────────────────────────────────────────────────────────────────────────────

def vbench_batch(
    vbench_info_json=None,
    output_dir='results_vbench/videos',
    image_types='indoor,scenery',
    resolution='1-1',
    crop_dir=None,
    fps=24,
    step=None,
    cfg_scale=None,
    mid_t=None,
    panogen_steps=None,
    panogen_seed=None,
    interactive=False,
    num_frames=161,
    step_deg=5.0,
    frame_width=720,
    frame_height=960,
    num_seeds=5,
    seed_start=0,
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

    # parse allowed image types (fire may deliver comma-separated value as a tuple)
    if isinstance(image_types, (list, tuple)):
        image_types = ','.join(image_types)
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
    _vbench_yaml = str(_WORLDFM_ROOT / 'vbench.yaml')
    cfg = _p.OmegaConf.load(_vbench_yaml)
    import random
    _pan_seed = panogen_seed if panogen_seed is not None else random.randint(0, 2**31 - 1)
    print(f'[vbench] panogen_seed={_pan_seed}')

    wfm_overrides = {k: v for k, v in [('step', step), ('cfg_scale', cfg_scale), ('mid_t', mid_t)] if v is not None}
    pan_overrides  = {k: v for k, v in [('num_inference_steps', panogen_steps), ('seed', _pan_seed)] if v is not None}
    if wfm_overrides or pan_overrides:
        patch = {}
        if wfm_overrides: patch['worldfm'] = wfm_overrides
        if pan_overrides:  patch['panogen'] = pan_overrides
        cfg = _p.OmegaConf.merge(cfg, _p.OmegaConf.create(patch))

    _p.setup_external_repos(
        hw_path=str(cfg.submodules.hw_path),
        moge_path=str(cfg.submodules.moge_path),
    )
    print('[vbench] loading panogen model...')
    panogen_demo = _p.step1_init(cfg=cfg)
    print('[vbench] loading WorldFM inference service...')
    svc, wcfg = _p.step4_init(cfg=cfg)
    print('[vbench] WorldFM ready\n')

    _render_size = int(cfg.render.render_size)
    K = _make_K(_render_size)

    generated = errors = skipped = 0
    t_start = time.time()

    def _step(label):
        print(f'  [{label}]', end=' ', flush=True)

    def _finish_step(t):
        print(f'{round(time.time()-t, 1)}s', flush=True)

    def _log_done():
        elapsed_m = round((time.time() - t_start) / 60, 1)
        print(f'\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_m}m')
        print(f'[vbench] stats -> {stats_path}')

    try:
        seeds = list(range(seed_start, seed_start + num_seeds))

        for ti, img in enumerate(images):
            import shutil
            image_path = os.path.join(image_dir, img['file'])
            stem       = Path(img['file']).stem

            if not os.path.isfile(image_path):
                print(f'[vbench] skip {ti+1}/{total}: image not found - {image_path}')
                for sd in seeds:
                    stats_w.writerow([ti, img['file'], img['type'], '', '', '', '', 'missing'])
                stats_f.flush()
                continue

            out_mp4s = [os.path.join(out_dir, f'{stem}_s{sd}.mp4') for sd in seeds]
            if all(os.path.exists(p) for p in out_mp4s):
                print(f'[vbench] skip {ti+1}/{total}: all seeds done - {stem}')
                for sd, p in zip(seeds, out_mp4s):
                    stats_w.writerow([ti, img['file'], img['type'], '', '', '', p, 'skipped'])
                stats_f.flush()
                skipped += num_seeds
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
            tmp_out = _WORLDFM_ROOT / '_vbench_tmp' / stem
            try:
                tmp_out.mkdir(parents=True, exist_ok=True)

                _step('panogen')
                ts = time.time()
                try:
                    panorama_img = _p.step1_panogen(image_path, tmp_out, cfg=cfg, prompt=stem, demo=panogen_demo)
                except Exception as _e:
                    raise RuntimeError(f'panogen failed: {_e}') from _e
                _finish_step(ts)

                _step('moge')
                ts = time.time()
                try:
                    pp_result = _p.step2_moge_pipeline(panorama_img, tmp_out, cfg=cfg)
                except Exception as _e:
                    raise RuntimeError(f'moge failed: {_e}') from _e
                _finish_step(ts)

                _step('renderer')
                ts = time.time()
                try:
                    renderer, cond_db, rcfg, S = _p.step3_init(pp_result, cfg=cfg)
                except Exception as _e:
                    raise RuntimeError(f'renderer failed: {_e}') from _e
                _finish_step(ts)

                if interactive:
                    _key_held, _key_stop = _start_key_listener()
                    _yaw0, _pitch0 = 0.0, 0.0
                    _step_r = np.radians(step_deg)
                    print(f'  [interactive] WASD to steer  ({num_frames} frames, {step_deg}°/frame)')

                _total_frames = num_frames

                # Run render + inference for each seed (different trajectory per seed)
                for sd_idx, sd in enumerate(seeds):
                    out_mp4 = os.path.join(out_dir, f'{stem}_s{sd}.mp4')
                    if os.path.exists(out_mp4):
                        print(f'  [seed {sd}] already done, skipping')
                        stats_w.writerow([ti, img['file'], img['type'], '', '', '', out_mp4, 'skipped'])
                        skipped += 1
                        continue

                    if interactive:
                        _yaw, _pitch = _yaw0, _pitch0
                        c2w_list_sd = None   # built frame by frame below
                    else:
                        c2w_list_sd = _make_panorama_trajectory(
                            num_frames, sd_idx, len(seeds),
                        )

                    _step(f'render+infer seed={sd}')
                    ts = time.time()
                    frames = []
                    _frame_times = []
                    for fi in range(_total_frames):
                        _tf = time.time()
                        if interactive:
                            if 'a' in _key_held: _yaw   -= _step_r
                            if 'd' in _key_held: _yaw   += _step_r
                            if 'w' in _key_held: _pitch += _step_r
                            if 's' in _key_held: _pitch -= _step_r
                            _pitch = float(np.clip(_pitch, -np.pi / 2 + 0.05, np.pi / 2 - 0.05))
                            c2w = _c2w_from_yaw_pitch(_yaw, _pitch)
                        else:
                            c2w = c2w_list_sd[fi]
                        try:
                            render_u8, cond_nearest = _p.step3_render_one(
                                renderer, cond_db, pp_result, K, c2w,
                                rcfg=rcfg, render_size=S,
                            )
                        except Exception as _fe:
                            raise RuntimeError(f'seed {sd} frame {fi+1}/{_total_frames}: render failed') from _fe
                        try:
                            frame = _p.step4_infer_one(svc, render_u8, cond_nearest, wcfg=wcfg, seed=sd * 1000 + fi)
                        except Exception as _fe:
                            raise RuntimeError(f'seed {sd} frame {fi+1}/{_total_frames}: WorldFM inference failed') from _fe
                        frames.append(frame)
                        _frame_times.append(time.time() - _tf)
                        _avg = sum(_frame_times) / len(_frame_times)
                        _eta_s = int(_avg * (_total_frames - fi - 1))
                        _vram = f'  VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else ''
                        if interactive:
                            _keys_str = ''.join(k for k in ('w', 'a', 's', 'd') if k in _key_held) or '-'
                            print(f'\r  [infer {fi+1}/{_total_frames}  {_frame_times[-1]:.1f}s/frame  ETA {_eta_s}s{_vram}  yaw={np.degrees(_yaw):.0f}°  keys={_keys_str}]  ',
                                  end='', flush=True)
                        else:
                            print(f'\r  [infer {fi+1}/{_total_frames}  {_frame_times[-1]:.1f}s/frame  ETA {_eta_s}s{_vram}]  ',
                                  end='', flush=True)
                    print()
                    _finish_step(ts)

                    h, w = frames[0].shape[:2]
                    out_w = frame_width  or w
                    out_h = frame_height or h
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (out_w, out_h))
                    for fr in frames:
                        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        if (out_w, out_h) != (w, h):
                            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                        vw.write(bgr)
                    vw.release()

                    dur  = round(time.time() - t0, 2)
                    vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else ''
                    try:
                        import psutil
                        ram = round(psutil.Process().memory_info().rss / 1024**3, 2)
                    except Exception:
                        ram = ''

                    print(f'  done {dur}s  VRAM {vram}GB  RAM {ram}GB  -> {stem}_s{sd}.mp4')
                    stats_w.writerow([ti, img['file'], img['type'], dur, vram, ram, out_mp4, 'ok'])
                    generated += 1

                if interactive:
                    _key_stop()
                del renderer, cond_db
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as _exc:
                _cause = getattr(_exc, '__cause__', None)
                _is_oom = isinstance(_exc, torch.cuda.OutOfMemoryError) or isinstance(_cause, torch.cuda.OutOfMemoryError)
                if not _is_oom:
                    _exc_type = type(_exc).__name__
                    if _exc_type in ('GatedRepoError', 'RepositoryNotFoundError',
                                     'LocalEntryNotFoundError', 'EntryNotFoundError'):
                        _log_done()
                        raise
                    print(f'\n  [ERROR] {stem}:', file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    stats_w.writerow([ti, img['file'], img['type'], '', '', '', '', 'error'])
                    errors += 1
                else:
                    print(f'\n  [OOM] clearing cache and skipping {stem}', file=sys.stderr)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    stats_w.writerow([ti, img['file'], img['type'], '', '', '', '', 'oom'])
                    errors += 1

            finally:
                shutil.rmtree(tmp_out, ignore_errors=True)

            stats_f.flush()

    except KeyboardInterrupt:
        print('\n[vbench] interrupted by user')

    finally:
        stats_f.close()
        _log_done()


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
