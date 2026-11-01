"""Inspect hardware and measure training cost before choosing a compute tier."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT, load_config, save_json, pick_device, seed_everything, synchronize, timestamp
import argparse
import platform
import shutil
import subprocess
import time
import torch
import yaml
from bioalgo.model import build_model


def measure(device, resolution, steps, batch_size=32, channels_last=False, mixed=False):
    model = build_model({'unfreeze': 9}, pretrained=False).to(device)
    x = torch.rand(batch_size, 3, resolution, resolution, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    y = torch.arange(batch_size, device=device) % 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    losses = []
    for step in range(steps + 2):
        if step == 2:
            synchronize(device)
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite benchmark loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    elapsed = time.perf_counter() - started
    del model, optimizer, x, y
    if device.type == 'mps': torch.mps.empty_cache()
    return {'resolution': resolution, 'steps': steps, 'batch_size': batch_size, 'channels_last': channels_last, 'mixed_precision': mixed, 'seconds': elapsed, 'seconds_per_image': elapsed / (steps * batch_size), 'finite_loss': all(__import__('math').isfinite(x) for x in losses)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--skip-benchmark', action='store_true')
    args = parser.parse_args()
    seed_everything(42)
    device = pick_device()
    info = {'timestamp': timestamp(), 'platform': platform.platform(), 'python': platform.python_version(), 'architecture': platform.machine(), 'torch': torch.__version__, 'device': str(device), 'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available(), 'free_disk_gb': shutil.disk_usage(ROOT).free / 1024**3, 'benchmarks': []}
    if device.type == 'cuda': info['gpu'] = torch.cuda.get_device_name(); info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if platform.system() == 'Darwin':
        for key, query in [('cpu', 'machdep.cpu.brand_string'), ('cpu_count', 'hw.ncpu'), ('memory_bytes', 'hw.memsize')]:
            result = subprocess.run(['sysctl', '-n', query], capture_output=True, text=True)
            info[key] = result.stdout.strip() if result.returncode == 0 else 'unavailable'
    if not args.skip_benchmark:
        for resolution in (112, 128, 160, 224):
            result = measure(device, resolution, args.steps)
            info['benchmarks'].append(result)
            save_json('results/system_info.json', info)
            print(f"{resolution}px: {result['seconds_per_image']:.4f} s/image", flush=True)
        for channels_last, mixed in ((True, False), (False, True)):
            if mixed and device.type == 'cpu': continue
            try: info['benchmarks'].append(measure(device, 128, min(args.steps, 10), channels_last=channels_last, mixed=mixed))
            except RuntimeError as exc: info['benchmarks'].append({'resolution':128, 'channels_last':channels_last,'mixed_precision':mixed,'error':str(exc)})
        times = {r['resolution']: r['seconds_per_image'] for r in info['benchmarks'] if not r.get('channels_last') and not r.get('mixed_precision') and 'seconds_per_image' in r}
        estimates = {}
        # Full-fine-tune measurements are conservative: some candidates freeze blocks.
        for name, res, fraction, epochs, budget in [('A',160,.3,3,110),('B',128,.2,3,72),('C',112,.1,2,42)]:
            train = 18000 * .7 * fraction * epochs * times[res]
            validation = 18000 * .15 * epochs * times[res] / 3
            estimates[name] = {'search_hours': 2 * budget * (train + validation) / 3600, 'final_training_hours': 9 * 15 * (18000 * .7 * times[224] + 18000 * .15 * times[224] / 3) / 3600}
        info['tier_estimates'] = estimates
        info['recommended_tier'] = next((name for name in ['A','B','C'] if estimates[name]['search_hours'] <= 3 and estimates[name]['final_training_hours'] <= 1.5), None)
        config=load_config()
        extended=bool(config.get('compute',{}).get('allow_extended_final_training',False))
        info['allow_extended_final_training']=extended
        info['selected_tier']=next((name for name in ['A','B','C'] if estimates[name]['search_hours']<=3 and (extended or estimates[name]['final_training_hours']<=1.5)),None)
        if info['selected_tier'] is not None:
            config['tier']=info['selected_tier']
            (ROOT/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
        info['selection_note']='Final-training cap relaxed by explicit compute configuration; search remains capped at three estimated hours.' if extended else 'Both original search and final-training time limits enforced.'
        info['measurement_note'] = 'Throughput-based planning estimate; synthetic tensors exclude data-loading/model-construction overhead; final freezing/early stopping can reduce training work.'
    save_json('results/system_info.json', info)
    print(f"System: {info['platform']}, PyTorch {info['torch']}, device {device}. Selected tier: {info.get('selected_tier', 'not benchmarked')}")
    if not args.skip_benchmark and info.get('selected_tier') is None:
        raise SystemExit('No tier meets the configured compute limits. Continue on the RTX 4050 Windows laptop, or explicitly adjust the compute policy before training.')

if __name__ == '__main__': main()
