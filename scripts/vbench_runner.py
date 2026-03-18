"""
vbench_runner.py — run WorldFM pipeline on every VBench i2v crop image.

For each image in the VBench JSON the runner:
  1. Writes a temp meta.json (image + shared K/c2w trajectory).
  2. Calls run_pipeline.py once (model loads once per image).
  3. Moves the output video to outputs/vbench/videos/.
  4. Appends a row to outputs/vbench/stats.csv.

Resume-safe: already-completed output videos are detected and skipped.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ALLOWED_TYPES = ['indoor', 'scenery']
NUM_FRAMES_FPS = 30   # fps passed to run_pipeline.py

# Default K and c2w trajectory taken from demo/meta.json (fov~90°, 512x512)
_DEMO_META = Path(__file__).resolve().parent.parent / 'demo' / 'meta.json'


# ---------- helpers ----------

def _load_demo_trajectory():
    """Return (K, c2w_list) from demo/meta.json as plain Python lists."""
    with open(_DEMO_META, encoding='utf-8') as f:
        m = json.load(f)
    return m['K'], m['c2w']


def _poll_vram(stop_event, readings):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True,
            ).strip().splitlines()[0].strip()
            if out.isdigit():
                readings.append(int(out))
        except Exception:
            pass
        time.sleep(5)


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().used / (1024 ** 3), 2)
    except Exception:
        return ''


def _vram_peak_gb(readings):
    return round(max(readings) / 1024.0, 2) if readings else ''


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vbench-json', required=True)
    ap.add_argument('--vbench-crop', required=True)
    ap.add_argument('--work-dir',    required=True)
    ap.add_argument('--fps',         type=int, default=NUM_FRAMES_FPS)
    args = ap.parse_args()

    work_dir   = Path(args.work_dir)
    out_base   = work_dir / 'outputs' / 'vbench' / 'videos'
    stats_path = work_dir / 'outputs' / 'vbench' / 'stats.csv'

    if not Path(args.vbench_json).exists():
        sys.exit(f'[vbench] ERROR: JSON not found: {args.vbench_json}')
    if not Path(args.vbench_crop).exists():
        sys.exit(f'[vbench] ERROR: crop dir not found: {args.vbench_crop}')

    K, c2w_list = _load_demo_trajectory()

    with open(args.vbench_json, encoding='utf-8') as f:
        entries = json.load(f)

    seen, images = set(), []
    for e in entries:
        fn = e.get('file_name', '')
        if fn in seen:
            continue
        if e.get('type') not in ALLOWED_TYPES:
            continue
        seen.add(fn)
        images.append({'file': fn, 'type': e['type']})

    total = len(images)
    print(f'\n=== VBench image list ({total} images, types: {", ".join(ALLOWED_TYPES)}) ===')
    for i, img in enumerate(images):
        img_ok = '  ' if (Path(args.vbench_crop) / img['file']).exists() else 'MISSING'
        print(f'  [{i+1:3}/{total}] ({img["type"]:<8}) {img_ok} {img["file"]}')
    print(f'=== {total} images ===\n')

    out_base.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats_is_new = not stats_path.exists()
    stats_f = open(stats_path, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(stats_f)
    if stats_is_new:
        writer.writerow(['task_idx', 'file', 'type', 'duration_s', 'vram_gb', 'ram_gb', 'out_path', 'status'])
        stats_f.flush()

    generated = errors = skipped = 0
    t_start = time.time()

    for ti, img in enumerate(images):
        img_src = Path(args.vbench_crop) / img['file']
        if not img_src.exists():
            print(f'[vbench] skip {ti+1}/{total}: image not found - {img_src}')
            writer.writerow([ti, img['file'], img['type'], '', '', '', '', 'missing'])
            stats_f.flush()
            continue

        stem = Path(img['file']).stem
        out_mp4 = out_base / f'{stem}.mp4'

        if out_mp4.exists():
            print(f'[vbench] skip {ti+1}/{total}: already done - {out_mp4.name}')
            writer.writerow([ti, img['file'], img['type'], '', '', '', str(out_mp4), 'skipped'])
            stats_f.flush()
            skipped += 1
            continue

        pct = round(100 * ti / total) if total else 0
        eta = ''
        done_so_far = generated + errors + skipped
        if done_so_far > 0:
            elapsed = time.time() - t_start
            rem = int(elapsed / done_so_far * (total - done_so_far))
            eta = f'  ETA {rem//3600:02d}h{(rem%3600)//60:02d}m{rem%60:02d}s'
        print(f'[vbench] [{ti+1}/{total}  {pct}%{eta}]  ({img["type"]})  {img["file"]}')

        # write temp meta.json
        tmp_dir  = work_dir / '_vbench_tmp'
        tmp_dir.mkdir(exist_ok=True)
        for f in tmp_dir.iterdir():
            f.unlink(missing_ok=True)
        shutil.copy(img_src, tmp_dir / img_src.name)

        meta = {
            'name':  stem,
            'image': img_src.name,
            'K':     K,
            'c2w':   c2w_list,
        }
        meta_path = tmp_dir / 'meta.json'
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

        vram_readings = []
        stop_evt    = threading.Event()
        vram_thread = threading.Thread(target=_poll_vram, args=(stop_evt, vram_readings), daemon=True)
        vram_thread.start()

        t0      = time.time()
        new_mp4 = None
        dur = fps_val = 0.0
        try:
            env = {**os.environ,
                   'PYTHONPATH': str(work_dir) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   'TOKENIZERS_PARALLELISM': 'false',
                   'TF_ENABLE_ONEDNN_OPTS':  '0'}
            subprocess.run([
                sys.executable, 'run_pipeline.py',
                '--meta',      str(meta_path),
                '--output_dir', str(tmp_dir / 'out'),
                '--save_mode', 'video',
                '--fps',       str(args.fps),
            ], cwd=str(work_dir), env=env, check=True)

            dur = round(time.time() - t0, 2)

            # run_pipeline.py saves to output_dir/<name>/output.mp4
            candidate = tmp_dir / 'out' / stem / 'output.mp4'
            if candidate.exists():
                shutil.move(str(candidate), str(out_mp4))
                new_mp4 = out_mp4
            else:
                # fallback: any mp4 produced under out/
                mp4s = list((tmp_dir / 'out').rglob('*.mp4'))
                if mp4s:
                    shutil.move(str(mp4s[0]), str(out_mp4))
                    new_mp4 = out_mp4
                else:
                    print(f'  WARNING: no output mp4 found')

        except Exception as exc:
            print(f'  EXCEPTION: {exc}', file=sys.stderr)
        finally:
            stop_evt.set()
            vram_thread.join(timeout=10)
            vram = _vram_peak_gb(vram_readings)
            ram  = _ram_gb()

        if new_mp4:
            print(f'  done in {dur}s  VRAM {vram}GB  RAM {ram}GB  -> {new_mp4.name}')
            writer.writerow([ti, img['file'], img['type'], dur, vram, ram, str(new_mp4), 'ok'])
            generated += 1
        else:
            writer.writerow([ti, img['file'], img['type'], '', '', '', '', 'error'])
            errors += 1
            print(f'[vbench] ERROR {img["file"]} — continuing')
        stats_f.flush()

        shutil.rmtree(tmp_dir, ignore_errors=True)

    stats_f.close()
    elapsed_m = round((time.time() - t_start) / 60, 1)
    print(f'\n[vbench] done - generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_m}m')
    print(f'[vbench] stats -> {stats_path}')


if __name__ == '__main__':
    main()
