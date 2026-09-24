#!/usr/bin/env python3
from .whisper_online_main import *
from .whisper_online_main import _events, format_text_lines

import sys
import argparse
import os
import logging
import json
import numpy as np
import socket

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

from simulstreaming.utils.server_utils import Connection


import io
import soundfile
import torch

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk, out_txt: bool, device="cuda"):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.out_txt = out_txt
        self.device = torch.device(device)

        self.is_first = True

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def send_result(self, iteration_output):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        events = _events(iteration_output)
        if events:
            for event in events:
                if self.out_txt:
                    for message in format_text_lines(event):
                        print(message, flush=True, file=sys.stderr)
                        self.connection.send(message)
                else:
                    message = json.dumps(event)
                    print(message, flush=True, file=sys.stderr)
                    self.connection.send(message)
        else:
            logger.debug("No text in this segment")

    def process(self):
        # handle one client connection
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.online_asr_proc.init()
        beg_time = time.time()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            for event in _events(o):
                event["emission_time"] = time.time() - beg_time
            try:
                self.send_result(o)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                logger.info("broken pipe -- connection closed?")
                break

        # The orchestration client half-closes its write side at end-of-stream
        # and keeps reading. Flush Whisper's remaining buffered audio so short
        # files and clean publisher shutdowns do not lose the final phrase.
        o = self.online_asr_proc.finish()
        for event in _events(o):
            event["emission_time"] = time.time() - beg_time
        try:
            self.send_result(o)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("client closed before the final result was delivered")
        log_cuda_peak("After stream", self.device)

def main_server(factory, add_args):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. "
            "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args,logger)

    # setting whisper object by args 


    asr, online = asr_factory(args, factory)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # warm up the ASR because the very first transcribe takes more time than the others. 
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    msg = "Whisper warmup uses one second of silence because no --warmup-file was supplied."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file,0,1)
            asr.warmup(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. "+msg)
            sys.exit(1)
    else:
        asr.warmup(np.zeros(SAMPLING_RATE, dtype=np.float32))
        logger.info(msg)

    log_cuda_memory(
        f"Models ready (device={args.device}, Whisper dtype={asr.model.model.dtype})",
        args.device,
    )

    # server loop

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info('Listening on'+str((args.host, args.port)))
        while True:
            conn, addr = s.accept()
            logger.info('Connected to client on {}'.format(addr))
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, min_chunk, args.out_txt, args.device)
            proc.process()
            conn.close()
            logger.info('Connection to client closed')
    logger.info('Connection closed, terminating.')
