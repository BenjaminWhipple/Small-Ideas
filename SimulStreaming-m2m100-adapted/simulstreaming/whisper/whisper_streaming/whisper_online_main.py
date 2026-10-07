#!/usr/bin/env python3

# This code is retrieved from the original WhisperStreaming whisper_online.py .
# It is refactored and simplified. Only the code that is needed for the 
# SimulWhisper backend is kept. 


import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
import json
from pathlib import Path
import torch


logger = logging.getLogger(__name__)


def _events(result):
    return result if isinstance(result, list) else ([result] if result else [])


def text_segments(event):
    """Group attributed words for the human-readable text output."""
    segments = []
    for word in event.get("words", []):
        speaker = word.get("speaker")
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["end"] = word["end"]
            segments[-1]["text"] += word["text"]
        else:
            segments.append({
                "start": word["start"], "end": word["end"],
                "speaker": speaker, "text": word["text"],
            })
    return segments


def format_text_lines(event, emission_time=None):
    """Format legacy or diarized text output for file and server entry points."""
    if "text" not in event:
        return []
    segments = text_segments(event) if event.get("speaker_segments") else [event]
    lines = []
    for segment in segments:
        fields = []
        if emission_time is not None:
            fields.append(f"{emission_time * 1000:.4f}")
        fields.extend((f"{segment['start'] * 1000:.0f}", f"{segment['end'] * 1000:.0f}"))
        if segment.get("speaker") is not None:
            fields.append(segment["speaker"])
        fields.append(segment["text"])
        lines.append(" ".join(fields))
    return lines

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

def processor_args(parser):
    """shared args for the online processors
    parser: argparse.ArgumentParser object
    """
    group = parser.add_argument_group("WhisperStreaming processor arguments (shared for simulation from file and for the server)")
    group.add_argument('--min-chunk-size', type=float, default=1.2, 
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter '
                        'time, it waits, otherwise it processes the whole segment that was received by this time.')

    group.add_argument('--lan', '--language', type=str, default="en", 
                        help="Source language code, e.g. en, de, cs, or auto for automatic language detection from speech.")
    group.add_argument('--task', type=str, default='transcribe', 
                        choices=["transcribe","translate"],
                        help="Transcribe or translate.")

    group.add_argument('--vac', action="store_true", default=False, 
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    group.add_argument('--vac-chunk-size', type=float, default=0.04, 
                        help='VAC sample size in seconds.')
    group.add_argument('--diarize', action='store_true',
                        help='Add connection-local ECAPA speaker labels (requires --vac and CUDA).')
    default_ecapa = Path(__file__).resolve().parents[4] / 'Diarization-Model' / 'exps' / 'pretrain.model'
    group.add_argument('--diarization-model-path', default=str(default_ecapa),
                        help='ECAPA-TDNN checkpoint containing speaker_encoder.* weights.')
    group.add_argument('--speaker-window', type=float, default=2.0)
    group.add_argument('--speaker-hop', type=float, default=0.5)
    group.add_argument('--speaker-threshold', type=float, default=0.30,
                        help='Cosine-similarity threshold for creating a new speaker.')
    group.add_argument('--max-speakers', type=int, default=0,
                        help='Maximum anonymous speakers; 0 is unlimited.')

    parser.add_argument("-l", "--log-level", dest="log_level", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the log level", default='DEBUG')

    parser.add_argument("--logdir", help="Directory to save audio segments and generated texts for debugging.",
                       default=None)
    parser.add_argument("--out-txt", action="store_true", help="Output formatted as not as jsonl but simple space-separated text: beg, end, text", 
                        default=False)

def asr_factory(args, factory=None):
    """
    Creates and configures an asr and online processor object through factory that is implemented in the backend.
    """
#    if backend is None:
#        backend = args.backend
#    if backend == "simul-whisper":
#        from simul_whisper_backend import simul_asr_factory
    if args.diarize and not args.vac:
        raise ValueError("--diarize requires --vac")
    if args.max_speakers < 0:
        raise ValueError("--max-speakers must be zero or positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; CPU fallback is disabled")
    if args.diarize and not Path(args.diarization_model_path).is_file():
        raise FileNotFoundError(
            f"ECAPA checkpoint not found: {args.diarization_model_path}"
        )

    asr, online = factory(args)

    if args.diarize:
        if device.type != "cuda":
            raise RuntimeError("--diarize requires a CUDA --device")
        from simulstreaming.diarization import ECAPAEncoder, DiarizationOnlineASRProcessor
        encoder = ECAPAEncoder(args.diarization_model_path, device=device)
        logger.info(
            "ECAPA ready: device=%s dtype=%s",
            next(encoder.model.parameters()).device,
            next(encoder.model.parameters()).dtype,
        )
        online = DiarizationOnlineASRProcessor(
            online, encoder, window=args.speaker_window, hop=args.speaker_hop,
            threshold=args.speaker_threshold, max_speakers=args.max_speakers,
        )

    # Create the OnlineASRProcessor
    if args.vac:
        from .vac_online_processor import VACOnlineASRProcessor
        online = VACOnlineASRProcessor(args.min_chunk_size, online, device=device)

    if args.task == "translate":
        if args.model_path.endswith(".en.pt"):
            logger.error(f"The model {args.model_path} is English only. Translation is not available. Terminating.")
            sys.exit(1)
        asr.set_translate_task()

    if hasattr(online, "warmup"):
        online.warmup()

    return asr, online


def log_cuda_memory(prefix, device):
    device = torch.device(device)
    if device.type != "cuda":
        return
    gib = 1024 ** 3
    logger.info(
        "%s CUDA %s: allocated=%.3f GiB reserved=%.3f GiB",
        prefix, device, torch.cuda.memory_allocated(device) / gib,
        torch.cuda.memory_reserved(device) / gib,
    )


def log_cuda_peak(prefix, device):
    device = torch.device(device)
    if device.type != "cuda":
        return
    gib = 1024 ** 3
    logger.info(
        "%s CUDA %s: allocated=%.3f GiB reserved=%.3f GiB "
        "peak_allocated=%.3f GiB peak_reserved=%.3f GiB",
        prefix, device,
        torch.cuda.memory_allocated(device) / gib,
        torch.cuda.memory_reserved(device) / gib,
        torch.cuda.max_memory_allocated(device) / gib,
        torch.cuda.max_memory_reserved(device) / gib,
    )

def set_logging(args,logger):
    logging.basicConfig(
        # this format would include module name:
        #    format='%(levelname)s\t%(name)s\t%(message)s')
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger().setLevel(args.log_level)
    logging.getLogger("simul_whisper").setLevel(args.log_level)
    logging.getLogger("whisper_streaming").setLevel(args.log_level)


def simulation_args(parser):
    simulation_group = parser.add_argument_group("Arguments for simulation from file")
    simulation_group.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    simulation_group.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    # TODO: offline mode is not implemented in SimulStreaming yet
#    simulation_group.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    simulation_group.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')

def main_simulation_from_file(factory, add_args=None):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''

    import argparse
    parser = argparse.ArgumentParser()

    processor_args(parser)
    if add_args is not None:
        add_args(parser)

    simulation_args(parser)

    args = parser.parse_args()
    args.offline = False  # TODO: offline mode is not implemented in SimulStreaming yet

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    set_logging(args,logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, factory)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path,0,1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.warmup(a)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    log_cuda_memory(
        f"Models ready (Whisper dtype={asr.model.model.dtype})", args.device
    )

    beg = args.start_at
    start = time.time()-beg

    def output_txt_transcript(iteration_output, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start

        events = _events(iteration_output)
        for event in events:
            for line in format_text_lines(event, emission_time=now):
                logger.debug(line)
                print(line, flush=True)
        if not events:
            logger.debug("No text in this segment")

    def output_json_transcript(iteration_output, now=None):
        if now is None:
            now = time.time() - start
        events = _events(iteration_output)
        for event in events:
            event['emission_time'] = now
            jline = json.dumps(event)
            logger.debug(jline)
            print(jline, flush=True)
        if not events:
            logger.debug("No text in this segment")

    if args.out_txt:
        output_transcript = output_txt_transcript
    else:
        output_transcript = output_json_transcript

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.info(f"## last processed {end:.2f}s")

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.info(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
    log_cuda_peak("After stream", args.device)
