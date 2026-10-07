import contextlib
import importlib.util
import io
from pathlib import Path
import socket
import tempfile
import threading
import unittest
import wave


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "stream_video.py"
SPEC = importlib.util.spec_from_file_location("stream_video", SCRIPT)
stream_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream_video)


class StreamVideoTests(unittest.TestCase):
    def make_wav(self, directory, seconds=0.1):
        path = Path(directory) / "fixture.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * int(16000 * seconds))
        return path

    def test_command_extracts_expected_pcm_format(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_wav(directory)
            args = stream_video.parse_args([
                str(video), "--start-at", "1.5", "--duration", "2", "--fast",
            ])
            command = stream_video.build_ffmpeg_command(args)
        self.assertNotIn("-re", command)
        self.assertIn("pcm_s16le", command)
        self.assertIn("16000", command)
        self.assertEqual(command[-1], "pipe:1")

    def test_ffplay_command_is_compatible_with_ubuntu_ffplay(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_wav(directory)
            args = stream_video.parse_args([str(video), "--play"])
            command = stream_video.build_ffplay_command(args)
        self.assertNotIn("-nostdin", command)
        self.assertIn("-autoexit", command)

    def test_automatic_server_command_enables_vac_and_diarization(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_wav(directory)
            args = stream_video.parse_args([str(video)])
            command = stream_video.build_server_command(args)
        self.assertIn("--vac", command)
        self.assertIn("--diarize", command)
        self.assertIn("--whisper-dtype", command)

    def test_quiet_server_is_client_side_only(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_wav(directory)
            args = stream_video.parse_args([
                str(video), "--start-server", "--quiet-server",
            ])
            command = stream_video.build_server_command(args)
        self.assertTrue(args.quiet_server)
        self.assertNotIn("--quiet-server", command)

    def test_end_to_end_against_line_framed_fake_server(self):
        try:
            listener = socket.socket()
        except PermissionError as error:
            self.skipTest(f"local sockets are prohibited by the test sandbox: {error}")
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        received = bytearray()

        def serve():
            connection, _ = listener.accept()
            first = connection.recv(4096)
            received.extend(first)
            connection.sendall(b'{"text":" partial"}\n')
            while True:
                block = connection.recv(4096)
                if not block:
                    break
                received.extend(block)
            connection.sendall(b'{"is_final":true}\n')
            connection.close()
            listener.close()

        server = threading.Thread(target=serve)
        server.start()
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_wav(directory)
            args = stream_video.parse_args([
                str(video), "--host", "127.0.0.1", "--port", str(port),
                "--fast", "--connect-timeout", "2", "--response-timeout", "2",
            ])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = stream_video.run(args)
        server.join(timeout=2)

        self.assertEqual(result, 0)
        self.assertEqual(len(received), 3200)
        self.assertEqual(
            stdout.getvalue().splitlines(),
            ['{"text":" partial"}', '{"is_final":true}'],
        )


if __name__ == "__main__":
    unittest.main()
