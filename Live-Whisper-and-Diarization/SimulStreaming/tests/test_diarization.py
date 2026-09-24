import argparse
import unittest
from unittest import mock
import numpy as np
import torch

from simulstreaming.diarization.ecapa import ECAPA_TDNN, filter_speaker_encoder_state
from simulstreaming.diarization.online import (
    DiarizationOnlineASRProcessor,
    OnlineSpeakerClustering,
)
from simulstreaming.whisper.whisper_streaming.whisper_online_main import (
    asr_factory,
    format_text_lines,
    processor_args,
    text_segments,
)
from simulstreaming.whisper.whisper_streaming.vac_online_processor import VACOnlineASRProcessor
from simulstreaming.whisper.whisper_streaming.whisper_server import ServerProcessor


class FakeEncoder:
    def __init__(self, embeddings):
        self.embeddings = iter(embeddings)
        self.seen_lengths = []

    def embed(self, audio):
        self.seen_lengths.append(len(audio))
        return next(self.embeddings)

    def warmup(self, sample_count):
        pass


class FakeOnline:
    def __init__(self):
        self.results = []
        self.offsets = []
        self.audio = []

    def init(self, offset=None):
        self.offsets.append(offset)

    def insert_audio_chunk(self, audio):
        self.audio.append(audio)

    def process_iter(self):
        return self.results.pop(0) if self.results else {}

    def finish(self):
        return self.results.pop(0) if self.results else {}


def unit(index):
    value = torch.zeros(192)
    value[index] = 1
    return value


class CheckpointTests(unittest.TestCase):
    def test_filter_and_strict_upstream_load(self):
        model = ECAPA_TDNN(C=1024)
        checkpoint = {
            **{f"speaker_encoder.{key}": value for key, value in model.state_dict().items()},
            "speaker_loss.weight": torch.ones(1),
        }
        state = filter_speaker_encoder_state(checkpoint)
        self.assertNotIn("speaker_loss.weight", state)
        incompatible = model.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_filter_rejects_unrelated_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "speaker_encoder"):
            filter_speaker_encoder_state({"other.weight": torch.ones(1)})


class ClusteringTests(unittest.TestCase):
    def test_normalization_hysteresis_and_new_speaker(self):
        clustering = OnlineSpeakerClustering(threshold=0.30)
        first = clustering.add(0, 2, unit(0) * 4)
        self.assertEqual(first[0].speaker, 0)
        self.assertAlmostEqual(float(clustering.centroids[0].norm()), 1.0)
        self.assertEqual(clustering.add(0.5, 2.5, unit(1)), [])
        confirmed = clustering.add(1.0, 3.0, unit(1))
        self.assertEqual([item.speaker for item in confirmed], [1, 1])

    def test_switch_hysteresis_and_cap(self):
        clustering = OnlineSpeakerClustering(threshold=0.30, max_speakers=2)
        clustering.add(0, 2, unit(0))
        clustering.add(0.5, 2.5, unit(1))
        clustering.add(1, 3, unit(1))
        self.assertEqual(clustering.current, 1)
        self.assertEqual(clustering.add(1.5, 3.5, unit(0)), [])
        switched = clustering.add(2, 4, unit(0))
        self.assertEqual([item.speaker for item in switched], [0, 0])
        # At the cap, an unmatched embedding is assigned to an existing cluster.
        clustering.add(2.5, 4.5, unit(2))
        clustering.add(3, 5, unit(2))
        self.assertEqual(len(clustering.centroids), 2)

    def test_reset(self):
        clustering = OnlineSpeakerClustering()
        clustering.add(0, 2, unit(0))
        clustering.reset()
        self.assertEqual(clustering.centroids, [])
        self.assertIsNone(clustering.current)


class OnlineProcessorTests(unittest.TestCase):
    def test_buffer_label_and_final_flush_with_short_audio(self):
        inner = FakeOnline()
        inner.results = [{
            "start": 0.1, "end": 0.4, "text": " Hello there",
            "tokens": [1, 2],
            "words": [
                {"start": 0.1, "end": 0.2, "text": " Hello", "tokens": [1]},
                {"start": 0.3, "end": 0.4, "text": " there", "tokens": [2]},
            ],
        }]
        encoder = FakeEncoder([unit(0)])
        processor = DiarizationOnlineASRProcessor(inner, encoder, window=2.0, hop=0.5)
        processor.init(offset=10.0)
        processor.insert_audio_chunk(np.ones(8000, dtype=np.float32))
        self.assertEqual(processor.process_iter(), [])
        events = processor.finish()
        self.assertEqual(encoder.seen_lengths, [32000])
        self.assertEqual((processor.timeline[0].start, processor.timeline[0].end), (10.0, 10.5))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["speaker"], "SPEAKER_00")
        self.assertTrue(all(word["speaker"] == "SPEAKER_00" for word in events[0]["words"]))
        self.assertEqual(events[0]["speaker_segments"][0]["speaker"], "SPEAKER_00")

    def test_mixed_speaker_grouping(self):
        inner = FakeOnline()
        processor = DiarizationOnlineASRProcessor(inner, FakeEncoder([]))
        from simulstreaming.diarization.online import WindowDecision
        processor.timeline = [
            WindowDecision(0, 1, 0, 0.8),
            WindowDecision(1, 2, 1, 0.7),
        ]
        event = processor._attribute({
            "start": 0.1, "end": 1.9, "text": " A B", "tokens": [1, 2],
            "words": [
                {"start": 0.1, "end": 0.3, "text": " A", "tokens": [1]},
                {"start": 1.5, "end": 1.8, "text": " B", "tokens": [2]},
            ],
        })
        self.assertIsNone(event["speaker"])
        self.assertEqual([s["speaker"] for s in event["speaker_segments"]],
                         ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([s["text"] for s in text_segments(event)], [" A", " B"])


class ArgumentTests(unittest.TestCase):
    def test_diarize_requires_vac_before_factory(self):
        parser = argparse.ArgumentParser()
        processor_args(parser)
        args = parser.parse_args(["--diarize"])
        args.device = "cuda"
        args.model_path = "large-v3.pt"
        called = False

        def factory(_):
            nonlocal called
            called = True
            return None, None

        with self.assertRaisesRegex(ValueError, "requires --vac"):
            asr_factory(args, factory)
        self.assertFalse(called)

    def test_cuda_failure_does_not_call_factory(self):
        parser = argparse.ArgumentParser()
        processor_args(parser)
        args = parser.parse_args(["--diarize", "--vac"])
        args.device = "cuda"
        args.model_path = "large-v3.pt"
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                asr_factory(args, lambda _: self.fail("factory must not run"))

    def test_missing_checkpoint_fails_before_model_load(self):
        parser = argparse.ArgumentParser()
        processor_args(parser)
        args = parser.parse_args([
            "--diarize", "--vac", "--diarization-model-path", "/missing/ecapa.model",
        ])
        args.device = "cuda"
        args.model_path = "large-v3.pt"
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(FileNotFoundError, "ECAPA checkpoint"):
                asr_factory(args, lambda _: self.fail("factory must not run"))


class OutputContractTests(unittest.TestCase):
    def test_vac_drains_events_before_final_marker(self):
        class FinishOnline:
            def finish(self):
                return [{"text": " one"}, {"text": " two"}]

        processor = VACOnlineASRProcessor.__new__(VACOnlineASRProcessor)
        processor.online = FinishOnline()
        processor.current_online_chunk_buffer_size = 1
        processor.is_currently_final = True
        result = processor.finish()
        self.assertEqual([item["is_final"] for item in result], [False, False, True])

    def test_server_diarized_text_schema_splits_speakers(self):
        class Connection:
            def __init__(self):
                self.lines = []

            def send(self, line):
                self.lines.append(line)

        connection = Connection()
        server = ServerProcessor(connection, None, 0.04, True, device="cpu")
        event = {
            "speaker_segments": [{"speaker": "SPEAKER_00"}],
            "words": [
                {"start": 1.0, "end": 1.2, "text": " Hi", "speaker": "SPEAKER_00"},
                {"start": 1.3, "end": 1.5, "text": " there", "speaker": "SPEAKER_01"},
            ],
            "text": " Hi there",
        }
        server.send_result([event])
        self.assertEqual(connection.lines, [
            "1000 1200 SPEAKER_00  Hi",
            "1300 1500 SPEAKER_01  there",
        ])
        self.assertEqual(format_text_lines(event, emission_time=2.5), [
            "2500.0000 1000 1200 SPEAKER_00  Hi",
            "2500.0000 1300 1500 SPEAKER_01  there",
        ])


if __name__ == "__main__":
    unittest.main()
