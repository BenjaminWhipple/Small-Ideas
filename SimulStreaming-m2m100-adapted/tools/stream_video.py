#!/usr/bin/env python3
"""Replay a video as a real-time 16 kHz PCM stream to SimulStreaming."""

import argparse
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SIMULSTREAMING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SIMULSTREAMING_ROOT.parent


def build_ffmpeg_command(args):
    command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if not args.fast:
        command.append("-re")
    if args.start_at:
        command.extend(("-ss", str(args.start_at)))
    command.extend(("-i", str(args.video), "-map", "0:a:0"))
    if args.duration is not None:
        command.extend(("-t", str(args.duration)))
    command.extend((
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
    ))
    return command


def build_ffplay_command(args):
    command = [
        args.ffplay, "-hide_banner", "-loglevel", "warning",
        "-autoexit",
    ]
    if args.start_at:
        command.extend(("-ss", str(args.start_at)))
    if args.duration is not None:
        command.extend(("-t", str(args.duration)))
    if args.mute_playback:
        command.append("-an")
    command.append(str(args.video))
    return command


def build_server_command(args):
    return [
        sys.executable,
        str(SIMULSTREAMING_ROOT / "simulstreaming_whisper_server.py"),
        "--model_path", str(args.model_path),
        "--language", args.language,
        "--vac", "--diarize",
        "--diarization-model-path", str(args.diarization_model_path),
        "--device", args.device,
        "--whisper-dtype", args.whisper_dtype,
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.server_log_level,
    ]


def connect_with_retry(host, port, timeout, server=None):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=min(2.0, timeout))
        except OSError as error:
            last_error = error
            if server is not None and server.poll() is not None:
                raise RuntimeError(
                    f"the automatically started server exited with status {server.returncode}"
                ) from error
            time.sleep(0.25)
    raise ConnectionError(
        f"could not connect to SimulStreaming at {host}:{port} within {timeout:g}s"
    ) from last_error


def receive_output(sock, errors):
    """Print line-framed server output as soon as it arrives."""
    try:
        with sock.makefile("r", encoding="utf-8", errors="replace", newline="\n") as stream:
            for line in stream:
                print(line.rstrip("\r\n"), flush=True)
    except (OSError, ValueError) as error:
        errors.append(error)


def send_pcm(stream, sock, chunk_bytes):
    total = 0
    while True:
        data = stream.read(chunk_bytes)
        if not data:
            return total
        sock.sendall(data)
        total += len(data)


def terminate(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def relay_server_stderr(stream, quiet=False):
    """Show server diagnostics without duplicating JSON emitted by the client."""
    for line in stream:
        line = line.rstrip("\r\n")
        if not quiet and line and not line.lstrip().startswith("{"):
            print(f"[server] {line}", file=sys.stderr, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract a video's audio with FFmpeg, pace it like a live source, "
            "stream it to simulstreaming_whisper_server, and print live output."
        )
    )
    parser.add_argument("video", type=Path, help="Input video or audio file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument(
        "--start-server", action="store_true",
        help="Start a local VAC + diarization server for this replay and stop it afterward.",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=SIMULSTREAMING_ROOT / "models" / "large-v3.pt",
        help="Whisper checkpoint used with --start-server.",
    )
    parser.add_argument(
        "--diarization-model-path", type=Path,
        default=PROJECT_ROOT / "Diarization-Model" / "exps" / "pretrain.model",
        help="ECAPA checkpoint used with --start-server.",
    )
    parser.add_argument(
        "--language", default="en", help="Whisper language used with --start-server."
    )
    parser.add_argument(
        "--quiet-server", action="store_true",
        help=(
            "Hide diagnostics from an automatically started server while still "
            "printing its live JSON/text results."
        ),
    )
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument(
        "--whisper-dtype", choices=("float16", "float32"), default="float16",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--server-log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play the video in FFplay while its audio is sent to the server.",
    )
    parser.add_argument(
        "--mute-playback", action="store_true",
        help="Show the video without local audio (only meaningful with --play).",
    )
    parser.add_argument("--start-at", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--duration", type=float, metavar="SECONDS")
    parser.add_argument(
        "--fast", action="store_true",
        help="Send as quickly as FFmpeg can decode instead of real-time pacing.",
    )
    parser.add_argument(
        "--chunk-ms", type=float, default=40.0,
        help="PCM socket write size in milliseconds (default: 40).",
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=60.0,
        help="How long to retry while the server starts (default: 60 seconds).",
    )
    parser.add_argument(
        "--response-timeout", type=float, default=60.0,
        help="How long to wait for the final server response after EOF.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help=argparse.SUPPRESS)
    parser.add_argument("--ffplay", default="ffplay", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if not args.video.is_file():
        parser.error(f"input file does not exist: {args.video}")
    if args.start_at < 0:
        parser.error("--start-at must be non-negative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.connect_timeout <= 0 or args.response_timeout <= 0:
        parser.error("connection and response timeouts must be positive")
    if args.fast and args.play:
        parser.error("--fast cannot be combined with --play because they would desynchronize")
    if args.mute_playback and not args.play:
        parser.error("--mute-playback requires --play")
    if args.quiet_server and not args.start_server:
        parser.error("--quiet-server requires --start-server")
    if args.start_server and not args.model_path.is_file():
        parser.error(f"Whisper checkpoint does not exist: {args.model_path}")
    if args.start_server and not args.diarization_model_path.is_file():
        parser.error(f"ECAPA checkpoint does not exist: {args.diarization_model_path}")
    for executable in (args.ffmpeg, args.ffplay if args.play else None):
        if executable and shutil.which(executable) is None:
            parser.error(f"required executable was not found: {executable}")
    return args


def run(args):
    chunk_bytes = max(
        BYTES_PER_SAMPLE,
        int(SAMPLE_RATE * BYTES_PER_SAMPLE * args.chunk_ms / 1000),
    )
    # PCM16 writes must end on a complete sample.
    chunk_bytes -= chunk_bytes % BYTES_PER_SAMPLE

    sock = None
    server = None
    server_stderr_thread = None
    ffmpeg = None
    ffplay = None
    receiver = None
    receive_errors = []
    try:
        if args.start_server:
            print("[client] starting local diarization server", file=sys.stderr)
            server = subprocess.Popen(
                build_server_command(args), stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True,
            )
            server_stderr_thread = threading.Thread(
                target=relay_server_stderr,
                args=(server.stderr, args.quiet_server),
                daemon=True,
            )
            server_stderr_thread.start()

        print(f"[client] connecting to {args.host}:{args.port}", file=sys.stderr)
        sock = connect_with_retry(
            args.host, args.port, args.connect_timeout, server=server
        )
        sock.settimeout(None)
        print(
            f"[client] streaming {args.video} as mono 16 kHz PCM"
            + (" and opening FFplay" if args.play else ""),
            file=sys.stderr,
        )

        ffmpeg = subprocess.Popen(build_ffmpeg_command(args), stdout=subprocess.PIPE)
        if args.play:
            ffplay = subprocess.Popen(build_ffplay_command(args))
            time.sleep(0.25)
            if ffplay.poll() not in (None, 0):
                raise RuntimeError(f"FFplay exited with status {ffplay.returncode}")

        receiver = threading.Thread(
            target=receive_output, args=(sock, receive_errors), daemon=True
        )
        receiver.start()
        sent = send_pcm(ffmpeg.stdout, sock, chunk_bytes)
        ffmpeg.stdout.close()
        ffmpeg_code = ffmpeg.wait()
        sock.shutdown(socket.SHUT_WR)

        receiver.join(timeout=args.response_timeout)
        if receiver.is_alive():
            raise TimeoutError(
                f"server did not close the response within {args.response_timeout:g}s"
            )
        if receive_errors:
            raise ConnectionError(f"server response failed: {receive_errors[0]}")
        if ffmpeg_code:
            raise RuntimeError(f"FFmpeg exited with status {ffmpeg_code}")

        seconds = sent / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        print(f"[client] sent {seconds:.2f}s of audio; server stream complete", file=sys.stderr)
        return 0
    except (BrokenPipeError, ConnectionError, OSError, RuntimeError, TimeoutError) as error:
        print(f"[client] error: {error}", file=sys.stderr)
        if not args.start_server and isinstance(error, ConnectionError):
            print(
                "[client] hint: start simulstreaming_whisper_server.py first "
                "or add --start-server",
                file=sys.stderr,
            )
        return 1
    finally:
        try:
            if sock is not None:
                sock.close()
        finally:
            terminate(ffmpeg)
            terminate(ffplay)
            terminate(server)
            if server_stderr_thread is not None:
                server_stderr_thread.join(timeout=1)


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[client] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
