"""Bounded C1-C5 client for an owned localhost comparison server; never stops services."""
import concurrent.futures
import argparse
import datetime
import json
import hashlib
import pathlib
import statistics
import threading
import time
import urllib.request

BASE = 'http://127.0.0.1:18080'
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUT = pathlib.Path(__file__).parent / 'qwen_c1_c5_results.json'
PROMPT = (
    'Write a detailed practical guide for organizing a home library. '
    'Cover sorting books by subject, choosing shelves, making labels, recording a catalog, '
    'protecting books from sunlight and moisture, finding space for new books, and building '
    'a sustainable weekly maintenance routine. Include concrete examples for a household '
    'with mixed fiction, reference books, textbooks, and children\'s books. '
    'Use clear paragraphs with helpful details. Continue writing until the response limit.'
)
BODY = {
    'model': 'qwen-native-compare',
    'messages': [{'role': 'user', 'content': PROMPT}],
    'max_tokens': 256, 'temperature': 0, 'seed': 42,
    'stream': True, 'stream_options': {'include_usage': True},
    'chat_template_kwargs': {'enable_thinking': False},
    'ignore_eos': True, 'cache_prompt': False,
}


def get(path):
    with LOCAL_HTTP.open(BASE + path, timeout=10) as response:
        return json.load(response)


def run_request(barrier, batch_start, request_index):
    barrier.wait()
    start = time.perf_counter()
    record = {'index': request_index, 'start_offset_s': start - batch_start[0]}
    first = last = None
    chunks = 0
    try:
        req = urllib.request.Request(
            BASE + '/v1/chat/completions', data=json.dumps(BODY).encode(),
            headers={'Content-Type': 'application/json'})
        with LOCAL_HTTP.open(req, timeout=180) as response:
            record['http_status'] = response.status
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    record['received_done'] = True
                    break
                obj = json.loads(payload)
                if 'error' in obj:
                    raise RuntimeError(str(obj['error']))
                for choice in obj.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('reasoning_content'):
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                        chunks += 1
                    if choice.get('finish_reason'):
                        record['finish_reason'] = choice['finish_reason']
                for key in ['usage', 'timings', 'model', 'system_fingerprint', 'id']:
                    if key in obj:
                        record[key] = obj[key]
        end = time.perf_counter()
        record.update(elapsed_s=end-start, end_offset_s=end-batch_start[0],
                      ttft_s=(first-start) if first else None,
                      first_token_offset_s=(first-batch_start[0]) if first else None,
                      last_token_offset_s=(last-batch_start[0]) if last else None,
                      streamed_content_chunks=chunks)
        n = record.get('usage', {}).get('completion_tokens', 0)
        record['end_to_end_tps'] = n / (end-start)
        if first and last and last > first:
            record['client_decode_tps'] = (n-1) / (last-first)
        if n != BODY['max_tokens'] or not record.get('received_done'):
            record['error'] = 'Incomplete generation or token count mismatch'
    except Exception as exc:
        record.update(error=repr(exc), elapsed_s=time.perf_counter()-start)
    return record


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-slots', type=int, choices=(5,), default=5)
    args = parser.parse_args()
    OUT = pathlib.Path(args.output)
    if OUT.exists():
        parser.error('output exists; select a new filename')
    props = get('/props')
    if props.get('model_alias') != 'qwen-native-compare':
        raise RuntimeError('localhost model alias differs; refusing load')
    initial_slots = get('/slots')
    if len(initial_slots) != 5 or any(s.get('n_ctx') != 262144 for s in initial_slots):
        raise RuntimeError('comparison requires five slots with 262144 tokens each')
    if args.expected_slots is not None and props.get('total_slots') != args.expected_slots:
        raise SystemExit(f"Expected {args.expected_slots} slots; server reports {props.get('total_slots')}. No load sent.")
    report = {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'endpoint': BASE, 'request': BODY,
        'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Short public prompt, allocated 256K per slot; not five occupied 256K sequences.',
        'server': {k: props.get(k) for k in [
            'total_slots', 'model_alias', 'model_ftype', 'build_info']},
        'context': props['default_generation_settings']['n_ctx'],
        'models': get('/v1/models'), 'slots_before': get('/slots'), 'batches': [],
        'method': 'Three simultaneous bursts per C. No cached prompt; thinking disabled; '
                  '256 forced output tokens. Burst aggregate = sum API completion tokens / '
                  'wall time from synchronized release to last completion. Per-user effective '
                  'rate = burst aggregate / C. Server decode excludes prompt and queue.',
    }
    order = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1, 3, 5, 2, 4]
    for batch_index, concurrency in enumerate(order, 1):
        slots = get('/slots')
        if any(s.get('is_processing') for s in slots):
            report['stopped_reason'] = 'Existing server slot is busy; no additional load sent.'
            break
        print(f'BEGIN batch={batch_index}/15 C={concurrency}', flush=True)
        batch_start = [None]
        def released():
            batch_start[0] = time.perf_counter()
        barrier = threading.Barrier(concurrency + 1, action=released)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_request, barrier, batch_start, i)
                       for i in range(concurrency)]
            barrier.wait()
            requests = [f.result() for f in futures]
        elapsed = max(r.get('end_offset_s', r['elapsed_s']) for r in requests)
        tokens = sum(r.get('usage', {}).get('completion_tokens', 0) for r in requests)
        batch = {'index': batch_index, 'concurrency': concurrency,
                 'elapsed_s': elapsed, 'completion_tokens': tokens,
                 'aggregate_tps': tokens/elapsed, 'requests': requests}
        events = []
        for request in requests:
            if request.get('first_token_offset_s') is not None:
                events.extend([(request['first_token_offset_s'], 1),
                               (request['last_token_offset_s'], -1)])
        active = peak = 0
        for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
            active += delta
            peak = max(peak, active)
        batch['max_overlapping_streams'] = peak
        report['batches'].append(batch)
        OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'batch': batch_index, 'c': concurrency,
                          'elapsed_s': round(elapsed, 3), 'tokens': tokens,
                          'aggregate_tps': round(tokens/elapsed, 3),
                          'max_overlapping_streams': peak,
                          'decode_tps': [round(r.get('timings', {}).get('predicted_per_second', 0), 3) for r in requests],
                          'ttft_s': [round(r.get('ttft_s') or 0, 3) for r in requests],
                          'errors': [r.get('error') for r in requests if r.get('error')]}), flush=True)
        if any(r.get('error') for r in requests):
            report['stopped_reason'] = 'Request failure; stopped remaining load.'
            break
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['health_after'] = get('/health')
    report['slots_after'] = get('/slots')
    summaries = []
    for c in range(1, 6):
        batches = [b for b in report['batches'] if b['concurrency'] == c]
        all_requests = [r for b in batches for r in b['requests']]
        requests = [r for r in all_requests if not r.get('error')]
        if not batches:
            continue
        if not requests:
            summaries.append({'concurrency': c, 'batches': len(batches),
                              'requests': len(all_requests),
                              'errors': [r.get('error') for r in all_requests if r.get('error')]})
            continue
        aggregate = sum(b['completion_tokens'] for b in batches) / sum(b['elapsed_s'] for b in batches)
        summary = {'concurrency': c, 'batches': len(batches), 'requests': len(requests),
                   'aggregate_tps': aggregate, 'per_user_effective_tps': aggregate/c,
                   'mean_individual_request_tps': statistics.mean(r['end_to_end_tps'] for r in requests),
                   'mean_server_decode_tps': statistics.mean(r['timings']['predicted_per_second'] for r in requests),
                   'mean_client_stream_tps': statistics.mean(r['client_decode_tps'] for r in requests if 'client_decode_tps' in r),
                   'min_client_stream_tps': min(r['client_decode_tps'] for r in requests if 'client_decode_tps' in r),
                   'max_client_stream_tps': max(r['client_decode_tps'] for r in requests if 'client_decode_tps' in r),
                   'max_overlapping_streams': max(b['max_overlapping_streams'] for b in batches),
                   'median_ttft_s': statistics.median(r['ttft_s'] for r in requests),
                   'max_ttft_s': max(r['ttft_s'] for r in requests),
                   'median_request_elapsed_s': statistics.median(r['elapsed_s'] for r in requests),
                   'mean_burst_elapsed_s': statistics.mean(b['elapsed_s'] for b in batches),
                   'prompt_tokens': sorted(set(r['usage']['prompt_tokens'] for r in requests)),
                   'cached_tokens': sorted(set(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) for r in requests)),
                   'errors': [r.get('error') for r in all_requests if r.get('error')]}
        summaries.append(summary)
    report['summary'] = summaries
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('FINAL ' + json.dumps(summaries), flush=True)
    print('EVIDENCE ' + str(OUT), flush=True)
    if report.get('stopped_reason'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
