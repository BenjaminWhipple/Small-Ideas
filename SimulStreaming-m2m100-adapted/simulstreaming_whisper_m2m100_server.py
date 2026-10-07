#!/usr/bin/env python3
"""Single-socket SimulStreaming Whisper and M2M100 server."""

from simulstreaming_whisper import simul_asr_factory, simulwhisper_args
from simulstreaming.translate import (
    add_m2m100_args,
    build_m2m100_processor,
    validate_m2m100_server_args,
)
from simulstreaming.whisper.whisper_streaming.whisper_server import main_server


def combined_args(parser):
    simulwhisper_args(parser)
    add_m2m100_args(parser)


if __name__ == "__main__":
    main_server(
        simul_asr_factory,
        add_args=combined_args,
        validate_args=validate_m2m100_server_args,
        wrap_online=build_m2m100_processor,
    )
