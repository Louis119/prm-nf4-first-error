#!/usr/bin/env python3
"""Score the full encoded pool with the native readout (A: BF16 head -> FP32 softmax).
No A/B/C, no repeats; one config per invocation."""
import argparse, sys, time, json
from pathlib import Path
sys.path.insert(0, '/mnt/workspace/prm-readout-precision-20261110/code')
from readout_common import *

ROOT = Path('/mnt/workspace/prm-tiecheck-20261111')
BUCKETS = [256, 512, 1024, 2048, 4096]

def pack(rows_):
    rows_ = sorted(rows_, key=lambda e: e['n_tokens'])
    batches, i = [], 0
    while i < len(rows_):
        # take a run of similar lengths into a bucket
        n = len(rows_)
        for bs in (64, 48, 32, 16, 8):
            j = min(i + bs, n)
            mx = rows_[j-1]['n_tokens']
            bucket = next((b for b in BUCKETS if b >= mx), 4096)
            est = (j - i) * bucket
            if est <= 28 * 1024 and bucket <= 2048:
                batches.append((rows_[i:j], bucket)); i = j; break
        else:
            j = min(i + 8, n)
            mx = rows_[j-1]['n_tokens']
            bucket = next((b for b in BUCKETS if b >= mx), 4096)
            batches.append((rows_[i:j], bucket)); i = j
    return batches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--backbone', required=True, choices=['bf16', 'nf4'])
    ap.add_argument('--device', required=True, type=int)
    a = ap.parse_args()
    out = ROOT / 'full' / a.model / f'{a.backbone}.jsonl'
    if out.exists():
        raise SystemExit(f'REFUSING_TO_OVERWRITE {out}')
    torch = set_env(a.device)
    enc = load_encoded(a.model)
    rows_ = list(enc.values())
    batches = pack(rows_)
    print(f'{a.model} {a.backbone}: {len(rows_)} rows, {len(batches)} batches', flush=True)
    t0 = time.time()
    model, head_fp32, head_bias_fp32, meta = load_model(a.model, a.backbone, a.device)
    print(f'loaded in {time.time()-t0:.1f}s', flush=True)
    cand = CANDIDATES[a.model]; pad = PAD_ID[a.model]
    got = []; perf = []
    t1 = time.time()
    try:
        for bi, (batch, bucket) in enumerate(batches):
            ids, mask, pos, lengths, n = batch_tensors(batch, pad, a.device)
            with torch.inference_mode(), torch.autocast('npu', enabled=False):
                hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                     use_cache=False, return_dict=True).last_hidden_state
            for j, e in enumerate(batch):
                pj = [int(x) for x in e['positions']]
                hs = hidden[j, pj, :]
                z = model.lm_head(hs).float()[:, cand]
                p = torch.softmax(z, dim=-1)[:, 0].float()
                got.append({'id': e['id'], 'n_steps': len(pj),
                            'scores': p.cpu().tolist(), 'token_sha256': e['token_sha256']})
            perf.append({'batch': bi, 'size': len(batch), 'bucket': bucket, 'padded': n})
            if bi % 20 == 19:
                import gc; gc.collect(); torch.npu.empty_cache()
            if bi % 20 == 0:
                print(f'  progress {bi}/{len(batches)} {time.time()-t1:.0f}s', flush=True)
        write_rows(out, got)
        save(ROOT / 'state' / f'{a.model}_{a.backbone}.json',
             {'status': 'complete', 'rows': len(got), 'batches': len(perf),
              'seconds': time.time() - t1, 'sha256': sha(out), 'meta': meta,
              'perf': perf[:3] + perf[-3:]})
        print('DONE', out, len(got), f'{time.time()-t1:.0f}s', flush=True)
    finally:
        del model
        import gc; gc.collect(); torch.npu.empty_cache()

if __name__ == '__main__':
    main()
