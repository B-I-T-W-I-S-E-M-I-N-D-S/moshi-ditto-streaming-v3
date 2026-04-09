import re, os

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

s_moshi  = read('pipeline/streaming_moshi.py')
s_server = read('streaming_server.py')
s_ditto  = read('pipeline/ditto_stream_adapter.py')
s_sync   = read('pipeline/sync_types.py')

checks = []

checks.append(("streaming_moshi imports TaggedToken",
    'from pipeline.sync_types import TaggedToken' in s_moshi))

checks.append(("streaming_moshi yields (seq, pcm_bytes) tuple",
    'yield ("audio", (current_seq, pcm_f32.tobytes()))' in s_moshi))

checks.append(("server unpacks audio (seq, pcm_bytes)",
    'moshi_seq, pcm_bytes = payload' in s_server))

checks.append(("server imports seq_pack",
    'from pipeline.sync_types import TaggedToken, seq_pack' in s_server))

checks.append(("server sends 0x01 + seq + pcm",
    'b"\\x01" + seq_bytes + pcm_bytes' in s_server))

checks.append(("server sends 0x02 + seq + jpeg",
    'b"\\x02" + seq_bytes + jpeg' in s_server))

checks.append(("server uses run_coroutine_threadsafe (thread-safe frame queue)",
    'run_coroutine_threadsafe' in s_server))

checks.append(("server has adaptive bridge flush (wait_for timeout)",
    'token_queue.get(), timeout=wait_time' in s_server))

checks.append(("server has shared error_event",
    'error_event: asyncio.Event = asyncio.Event()' in s_server))

checks.append(("server has 10s cleanup timeout",
    'timeout=10.0' in s_server))

idx_patch  = s_ditto.find('self._patch_writer_worker_function()')
idx_setup  = s_ditto.find('self.sdk.setup(')
checks.append(("ditto pre-patches writer BEFORE sdk.setup()",
    idx_patch >= 0 and idx_setup >= 0 and idx_patch < idx_setup))

checks.append(("ditto writer uses non-blocking put_nowait",
    'frame_queue.put_nowait(jpeg)' in s_ditto))

checks.append(("ditto push_features has 2s queue timeout",
    'self.sdk.audio2motion_queue.put(features, timeout=2.0)' in s_ditto))

checks.append(("sync_types exports seq_pack",
    'def seq_pack' in s_sync))

all_ok = True
for label, result in checks:
    status = '  OK  ' if result else '  FAIL'
    if not result:
        all_ok = False
    print(f'{status}  {label}')

print()
print('All cross-references OK!' if all_ok else 'SOME REFERENCES FAILED')
