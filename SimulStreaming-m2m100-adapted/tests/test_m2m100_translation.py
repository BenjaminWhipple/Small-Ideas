import argparse
import json
from types import SimpleNamespace
import unittest

import torch

from simulstreaming.translate.m2m100 import (
    M2M100Translator,
    TranslationOnlineProcessor,
    TranslationProcessingError,
    validate_m2m100_server_args,
)
from simulstreaming.whisper.whisper_streaming.whisper_server import ServerProcessor


class FakeOnline:
    def __init__(self, process_results=None, finish_result=None):
        self.process_results = list(process_results or [])
        self.finish_result = finish_result if finish_result is not None else {}
        self.init_offsets = []
        self.audio = []

    def init(self, offset=None):
        self.init_offsets.append(offset)

    def insert_audio_chunk(self, audio):
        self.audio.append(audio)

    def process_iter(self):
        return self.process_results.pop(0) if self.process_results else {}

    def finish(self):
        return self.finish_result


class FakeTranslator:
    model_name = "facebook/m2m100_418M"
    pieces = {
        1: "Hello",
        2: " world",
        3: "!",
        4: " there",
        5: "New",
    }

    def __init__(self, hypotheses=None, supported=None, failure=None):
        self.hypotheses = list(hypotheses or [])
        self.supported = set(supported or {"en", "es", "fr"})
        self.failure = failure
        self.calls = []

    def supports_language(self, language):
        return language in self.supported

    def decode(self, tokens):
        return "".join(self.pieces[token] for token in tokens)

    def translate(self, source_text, source_language, target_language, forced_prefix=()):
        self.calls.append({
            "source_text": source_text,
            "source_language": source_language,
            "target_language": target_language,
            "forced_prefix": list(forced_prefix),
        })
        if self.failure is not None:
            raise self.failure
        return list(self.hypotheses.pop(0))


class TranslationProcessorTests(unittest.TestCase):
    def test_local_agreement_confirms_prefix_and_flushes_remainder(self):
        online = FakeOnline(
            process_results=[
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": " Hola",
                    "detected_language": "es",
                    "speaker": "SPEAKER_00",
                    "is_final": False,
                },
                {
                    "start": 0.5,
                    "end": 1.0,
                    "text": " mundo",
                    "detected_language": "es",
                    "is_final": False,
                },
            ],
            finish_result={"is_final": True},
        )
        translator = FakeTranslator([[1, 2], [1, 2, 3], [1, 2, 4]])
        processor = TranslationOnlineProcessor(online, translator, "auto", "en")

        first = processor.process_iter()
        second = processor.process_iter()
        final = processor.finish()

        self.assertEqual(first["speaker"], "SPEAKER_00")
        self.assertEqual(first["translation"]["text"], "")
        self.assertEqual(first["translation"]["unconfirmed_text"], "Hello world")
        self.assertEqual(second["translation"]["text"], "Hello world")
        self.assertEqual(second["translation"]["unconfirmed_text"], "!")
        self.assertEqual(final["translation"]["text"], " there")
        self.assertEqual(final["translation"]["unconfirmed_text"], "")
        self.assertEqual(final["translation"]["status"], "COMPLETE")
        self.assertEqual(
            [call["source_text"] for call in translator.calls],
            [" Hola", " Hola mundo", " Hola mundo"],
        )
        self.assertEqual(
            [call["forced_prefix"] for call in translator.calls],
            [[], [], [1, 2]],
        )

    def test_event_lists_keep_order_and_final_marker_gets_flush(self):
        online = FakeOnline(process_results=[[
            {"text": " Hola", "detected_language": "es", "is_final": False},
            {"is_final": True},
        ]])
        translator = FakeTranslator([[1, 2], [1, 2]])
        processor = TranslationOnlineProcessor(online, translator, "auto", "en")

        events = processor.process_iter()

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["translation"]["status"], "INCOMPLETE")
        self.assertEqual(events[1]["translation"]["status"], "COMPLETE")
        self.assertEqual(events[1]["translation"]["text"], "Hello world")

    def test_identity_translation_preserves_exact_fragments(self):
        online = FakeOnline(
            process_results=[
                {"text": "Hel", "is_final": False},
                {"text": "lo world", "is_final": False},
            ],
            finish_result={"is_final": True},
        )
        translator = FakeTranslator([])
        processor = TranslationOnlineProcessor(online, translator, "en", "en")

        first = processor.process_iter()["translation"]
        second = processor.process_iter()["translation"]
        final = processor.finish()["translation"]

        self.assertEqual((first["text"], second["text"]), ("Hel", "lo world"))
        self.assertEqual(first["model"], "identity")
        self.assertEqual(final["text"], "")
        self.assertEqual(final["status"], "COMPLETE")
        self.assertEqual(translator.calls, [])

    def test_unsupported_auto_language_reports_error_until_next_turn(self):
        online = FakeOnline(process_results=[
            {"text": " first", "detected_language": "xx", "is_final": False},
            {"text": " more", "detected_language": "xx", "is_final": False},
            {"is_final": True},
            {"text": " hola", "detected_language": "es", "is_final": False},
        ])
        translator = FakeTranslator([[1]], supported={"es", "en"})
        processor = TranslationOnlineProcessor(online, translator, "auto", "en")

        first = processor.process_iter()["translation"]
        second = processor.process_iter()["translation"]
        final = processor.process_iter()["translation"]
        recovered = processor.process_iter()["translation"]

        self.assertEqual(first["status"], "ERROR")
        self.assertEqual(first["error"]["code"], "unsupported_source_language")
        self.assertEqual(second["status"], "ERROR")
        self.assertEqual(final["status"], "ERROR")
        self.assertEqual(recovered["status"], "INCOMPLETE")
        self.assertEqual(len(translator.calls), 1)

    def test_inference_error_preserves_transcript_and_resets_on_init(self):
        event = {
            "text": " Hola",
            "detected_language": "es",
            "tokens": [1],
            "words": [{"text": " Hola"}],
            "is_final": False,
        }
        online = FakeOnline(process_results=[event])
        translator = FakeTranslator(
            failure=TranslationProcessingError("source_too_long", "too long")
        )
        processor = TranslationOnlineProcessor(online, translator, "auto", "en")

        result = processor.process_iter()
        processor.init()

        self.assertEqual(result["text"], event["text"])
        self.assertEqual(result["tokens"], event["tokens"])
        self.assertEqual(result["words"], event["words"])
        self.assertEqual(result["translation"]["error"]["code"], "source_too_long")
        self.assertEqual(processor.source_text, "")
        self.assertEqual(online.init_offsets, [None])

    def test_silence_only_final_marker_is_unchanged(self):
        online = FakeOnline(finish_result={"is_final": True})
        processor = TranslationOnlineProcessor(
            online, FakeTranslator([]), "auto", "en"
        )
        self.assertEqual(processor.finish(), {"is_final": True})


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.src_lang = None

    def __call__(self, text, return_tensors):
        return {
            "input_ids": torch.tensor([self.input_ids]),
            "attention_mask": torch.ones((1, len(self.input_ids)), dtype=torch.long),
        }

    def get_lang_id(self, language):
        return {"en": 100, "es": 101}[language]

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens)


class FakeGenerationModel:
    def __init__(self, generated):
        self.generated = generated
        self.config = SimpleNamespace(
            decoder_start_token_id=2,
            max_position_embeddings=16,
        )
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([self.generated])


class AdapterTests(unittest.TestCase):
    def make_adapter(self, generated, input_ids=(101, 10, 2)):
        adapter = M2M100Translator.__new__(M2M100Translator)
        adapter.model_name = "facebook/m2m100_418M"
        adapter.device = torch.device("cpu")
        adapter.tokenizer = FakeTokenizer(list(input_ids))
        adapter.model = FakeGenerationModel(generated)
        adapter.max_source_tokens = 16
        return adapter

    def test_generation_forces_target_language(self):
        adapter = self.make_adapter([2, 100, 7, 8, 2])
        result = adapter.translate("hola", "es", "en")
        call = adapter.model.calls[0]
        self.assertEqual(result, [7, 8])
        self.assertEqual(call["forced_bos_token_id"], 100)
        self.assertNotIn("decoder_input_ids", call)

    def test_generation_uses_confirmed_decoder_prefix(self):
        adapter = self.make_adapter([2, 100, 7, 8, 9, 2])
        result = adapter.translate("hola", "es", "en", forced_prefix=[7, 8])
        call = adapter.model.calls[0]
        self.assertEqual(result, [7, 8, 9])
        self.assertEqual(call["decoder_input_ids"].tolist(), [[2, 100, 7, 8]])
        self.assertNotIn("forced_bos_token_id", call)

    def test_source_limit_is_recoverable_error(self):
        adapter = self.make_adapter([2, 100, 7, 2], input_ids=range(17))
        with self.assertRaisesRegex(TranslationProcessingError, "source limit") as raised:
            adapter.translate("long", "es", "en")
        self.assertEqual(raised.exception.code, "source_too_long")

    def test_unterminated_generation_is_recoverable_error(self):
        adapter = self.make_adapter([2, 100, 7, 8])
        with self.assertRaisesRegex(TranslationProcessingError, "target-token limit") as raised:
            adapter.translate("hola", "es", "en")
        self.assertEqual(raised.exception.code, "target_too_long")


class ArgumentTests(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "vac": True,
            "task": "transcribe",
            "out_txt": False,
            "translation_target": "en",
            "translation_device": None,
            "translation_dtype": "auto",
            "lan": "es",
            "device": "cuda",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_valid_fixed_and_auto_source(self):
        validate_m2m100_server_args(self.args())
        validate_m2m100_server_args(self.args(lan="auto"))

    def test_requires_vac_transcription_and_json(self):
        for args, message in (
            (self.args(vac=False), "requires --vac"),
            (self.args(task="translate"), "requires --task transcribe"),
            (self.args(out_txt=True), "does not support --out-txt"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_m2m100_server_args(args)

    def test_rejects_languages_and_cpu_float16(self):
        with self.assertRaisesRegex(ValueError, "target language"):
            validate_m2m100_server_args(self.args(translation_target="xx"))
        with self.assertRaisesRegex(ValueError, "source language"):
            validate_m2m100_server_args(self.args(lan="xx"))
        with self.assertRaisesRegex(ValueError, "not supported on CPU"):
            validate_m2m100_server_args(
                self.args(
                    device="cpu", translation_device="cpu", translation_dtype="float16"
                )
            )
        with self.assertRaisesRegex(ValueError, "invalid --translation-device"):
            validate_m2m100_server_args(
                self.args(translation_device="not-a-device")
            )


class LegacyServerRegressionTests(unittest.TestCase):
    def test_legacy_json_output_has_no_translation_block(self):
        class Connection:
            def __init__(self):
                self.lines = []

            def send(self, line):
                self.lines.append(line)

        connection = Connection()
        server = ServerProcessor(connection, None, 0.04, False, device="cpu")
        server.send_result({"start": 0.0, "end": 1.0, "text": " hello"})
        result = json.loads(connection.lines[0])
        self.assertEqual(result["text"], " hello")
        self.assertNotIn("translation", result)


if __name__ == "__main__":
    unittest.main()
