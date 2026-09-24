# SimulStreaming Whisper with online diarization

This checkout contains the Whisper portion of SimulStreaming. It supports live
or file-based 16 kHz mono transcription, Whisper's built-in `--task translate`
mode, Silero voice activity control (VAC), and optional online ECAPA-TDNN
speaker diarization.

The diarized processing path is:

```text
16 kHz PCM → GPU Silero VAC → GPU ECAPA-TDNN → GPU Whisper → labeled events
```

Socket and audio-framing work remains CPU-side. Neural inference uses one
explicit Torch device and never silently falls back from CUDA to CPU.

## Installation

Install the runtime dependencies in the environment used to launch the tools:

```bash
pip install -r requirements_whisper.txt
```

Compared with the original SimulStreaming Whisper runtime, this integration
adds no new neural-framework package: ECAPA reuses `torch` and `torchaudio`, and
the online adapter uses `numpy`. `soundfile` is now listed explicitly because
the TCP server imports it directly (it was commonly installed transitively by
`librosa`). Do not install `Diarization-Model/requirements.txt` for serving; it
belongs to the upstream training project and pins an obsolete Torch/CUDA stack.

Non-Python/runtime requirements are:

- an NVIDIA CUDA-capable PyTorch installation and GPU;
- FFmpeg for video/audio decoding;
- FFplay only when using `tools/stream_video.py --play`;
- the local Whisper large-v3 checkpoint (about 2.9 GiB);
- the ECAPA checkpoint (about 64 MiB); and
- network access on the first Silero VAD load, unless its Torch Hub files are
  already cached.

The default ECAPA checkpoint path is
`../Diarization-Model/exps/pretrain.model`. The runtime architecture in
`simulstreaming/diarization/ecapa.py` is adapted from
[TaoRuijie's ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN), under its MIT
license. `Diarization-Model/` is retained as upstream training and reference
material; it is not imported by the live service.

## File simulation

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n multimodalsensorfusion python \
  simulstreaming_whisper.py input-16khz-mono.wav \
  --model_path models/large-v3.pt --language en --vac --diarize \
  --comp_unaware
```

## TCP server

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n multimodalsensorfusion python \
  simulstreaming_whisper_server.py \
  --model_path models/large-v3.pt --language en --vac --diarize \
  --host localhost --port 43007
```

The server accepts mono, signed 16-bit, little-endian 16 kHz PCM. It handles one
connection at a time. Anonymous speaker identities and their centroids reset at
the beginning of every connection.

### Replay a video like a live source

With the server running, the video client asks FFmpeg to decode and pace the
audio in real time, streams 40 ms PCM chunks, and prints every server line as it
arrives:

```bash
python tools/stream_video.py /path/to/video.webm
```

For a single-command local run, let the client start and stop the GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/stream_video.py /path/to/video.webm \
  --start-server
```

Add `--play` to open the same file in FFplay while it is streamed, which makes
it convenient to compare speaker labels and text with the visible speakers:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/stream_video.py /path/to/video.webm \
  --start-server --play
```

Useful controls include `--start-at SECONDS`, `--duration SECONDS`, `--host`,
and `--port`. `--mute-playback` keeps the comparison window silent. `--fast`
disables real-time pacing for automated tests and cannot be combined with
`--play`, since those modes would intentionally desynchronize.

#### Local multi-speaker examples

The following excerpts produced two speaker identities in local tests. They run
at real-time speed and open the matching video for visual comparison.

For the shortest complete demonstration, run:

```bash
./demo_diarization.sh
```

Use `./demo_diarization.sh --no-play` on a headless machine.
The demo passes `--quiet-server`, so server diagnostics are hidden while live
JSON results remain visible. Omit that option when debugging model startup.

English interviewer and guest (`SPEAKER_01` asks a question, followed by
`SPEAKER_00`):

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output \
  -n multimodalsensorfusion \
  python tools/stream_video.py \
  "../data/English/Interview with Jimmy Wales at Wikimania 2025 Nairobi.webm" \
  --start-server --play --language en --start-at 90 --duration 35
```

Mandarin host and guest (two identities in the opening exchange):

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output \
  -n multimodalsensorfusion \
  python tools/stream_video.py \
  "../data/Mandarin Chinese/2023年4月7日 【星访谈】专访冯小刚：创作者需要有生活 不能被“绑架”.webm" \
  --start-server --play --language zh --start-at 7 --duration 45
```

## Diarization and GPU options

- `--diarize`: enable speaker attribution; requires `--vac` and CUDA.
- `--diarization-model-path PATH`: ECAPA checkpoint containing
  `speaker_encoder.*` weights.
- `--speaker-window 2.0`: embedding window in seconds.
- `--speaker-hop 0.5`: embedding hop in seconds.
- `--speaker-threshold 0.30`: cosine-similarity boundary below which a new
  speaker is proposed. This is data-dependent and should be tuned on production
  audio.
- `--max-speakers 0`: speaker cap; zero is unlimited. At a positive cap,
  unmatched audio is assigned to the closest existing speaker.
- `--device cuda`: shared Torch device. CUDA availability is checked at startup.
- `--whisper-dtype {float16,float32}`: Whisper dtype, default `float16`.

Silero and ECAPA run in FP32; Whisper defaults to FP16. Models are warmed before
the server starts listening. Startup and per-stream logs report allocated,
reserved, and peak CUDA memory.

Speaker creation and changes require two consecutive windows. This adds roughly
2–2.5 seconds of attribution latency: transcript events are buffered until their
speaker labels are stable. Short final turns are wrap-padded for ECAPA and
flushed. Already-emitted labels are never revised.

Overlapping or ambiguous speech receives the closest single existing speaker.
Multi-speaker overlap output, enrollment, and named-speaker recognition are not
supported.

## JSON output

Non-diarized output is unchanged. With `--diarize`, transcript objects retain
their existing fields and add:

```json
{
  "start": 1.2,
  "end": 3.4,
  "text": " Hello world",
  "speaker": "SPEAKER_00",
  "speaker_segments": [
    {
      "start": 1.2,
      "end": 3.4,
      "speaker": "SPEAKER_00",
      "similarity": 0.61
    }
  ],
  "words": [
    {
      "start": 1.2,
      "end": 1.5,
      "text": " Hello",
      "tokens": [123],
      "speaker": "SPEAKER_00"
    }
  ]
}
```

Top-level `speaker` is `null` when one event contains multiple speakers.
`speaker_segments` groups adjacent words with the same label. `similarity` is a
cosine similarity to an online centroid, not a calibrated confidence score.
Buffered transcript events are emitted before the existing `{"is_final": true}`
turn marker.

With `--out-txt`, mixed events are split at speaker boundaries:

- Server: `start_ms end_ms SPEAKER_00 text`
- File simulation: `emission_ms start_ms end_ms SPEAKER_00 text`

## Tests

Run the CPU-safe clustering, checkpoint-filtering, buffering, and output tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Real-model tests require the project Conda environment, GPU 0, the local
large-v3 checkpoint, the ECAPA checkpoint, and benchmark media. Those assets are
kept outside normal Git tracking where configured by the repository `.gitignore`.

## Attribution

The Whisper streaming implementation retains code adapted from SimulStreaming,
Simul-Whisper, WhisperStreaming, OpenAI Whisper, and Silero VAD. See source-file
headers and `LICENCE.txt` for the corresponding notices.
