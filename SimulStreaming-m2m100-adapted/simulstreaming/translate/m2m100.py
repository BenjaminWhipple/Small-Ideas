"""Incremental M2M100 translation for Whisper JSON events."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import torch

from simulstreaming.whisper.whisper_streaming.base import OnlineProcessorInterface


logger = logging.getLogger(__name__)

DEFAULT_M2M100_MODEL = "facebook/m2m100_418M"

# Language identifiers published for facebook/m2m100_418M. Keeping this list
# local lets the combined server reject invalid fixed-language configurations
# before loading either neural model.
M2M100_LANGUAGE_CODES = frozenset(
    """
    af am ar ast az ba be bg bn br bs ca ceb cs cy da de el en es et fa ff fi
    fr fy ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kk km kn
    ko lb lg ln lo lt lv mg mk ml mn mr ms my ne nl no ns oc or pa pl ps pt
    ro ru sd si sk sl so sq sr ss su sv sw ta th tl tn tr uk ur uz vi wo xh
    yi yo zh zu
    """.split()
)


class TranslationProcessingError(RuntimeError):
    """A recoverable per-utterance translation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> list[int]:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return list(left[:length])


def _decoded_suffix(full_text: str, prefix_text: str, fallback: str) -> str:
    """Return a decoded suffix without losing SentencePiece whitespace."""
    if full_text.startswith(prefix_text):
        return full_text[len(prefix_text):]
    return fallback


class M2M100Translator:
    """Thin inference adapter around ``facebook/m2m100_418M``."""

    def __init__(
        self,
        model_name: str = DEFAULT_M2M100_MODEL,
        device: str | torch.device = "cuda",
        dtype: str = "auto",
    ):
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "M2M100 translation requires transformers and sentencepiece; "
                "install requirements_m2m100.txt"
            ) from error

        self.model_name = model_name
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA translation device {self.device} was requested but CUDA is unavailable"
            )

        if dtype == "auto":
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        if self.device.type == "cpu" and self.dtype == torch.float16:
            raise ValueError("float16 M2M100 inference is not supported on CPU")

        logger.info(
            "Loading M2M100 model %s on %s with %s",
            model_name,
            self.device,
            self.dtype,
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.max_source_tokens = int(
            getattr(self.model.config, "max_position_embeddings", 1024)
        )
        logger.info(
            "M2M100 ready: device=%s dtype=%s max_source_tokens=%d",
            self.device,
            self.dtype,
            self.max_source_tokens,
        )

    def supports_language(self, language: str) -> bool:
        return language in M2M100_LANGUAGE_CODES

    def decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(
            list(tokens),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def translate(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        forced_prefix: Sequence[int] = (),
    ) -> list[int]:
        if not self.supports_language(source_language):
            raise TranslationProcessingError(
                "unsupported_source_language",
                f"M2M100 does not support source language '{source_language}'",
            )
        if not self.supports_language(target_language):
            raise TranslationProcessingError(
                "unsupported_target_language",
                f"M2M100 does not support target language '{target_language}'",
            )

        self.tokenizer.src_lang = source_language
        encoded = self.tokenizer(source_text, return_tensors="pt")
        source_token_count = int(encoded["input_ids"].shape[-1])
        if source_token_count > self.max_source_tokens:
            raise TranslationProcessingError(
                "source_too_long",
                "The utterance exceeds M2M100's "
                f"{self.max_source_tokens}-token source limit",
            )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        target_language_id = self.tokenizer.get_lang_id(target_language)
        generation_args = {}
        decoder_prompt_length = 1
        if forced_prefix:
            decoder_ids = [
                self.model.config.decoder_start_token_id,
                target_language_id,
                *forced_prefix,
            ]
            decoder_prompt_length = len(decoder_ids)
            generation_args["decoder_input_ids"] = torch.tensor(
                [decoder_ids], dtype=torch.long, device=self.device
            )
        else:
            generation_args["forced_bos_token_id"] = target_language_id

        max_new_tokens = self.max_source_tokens - decoder_prompt_length
        if max_new_tokens < 1:
            raise TranslationProcessingError(
                "target_too_long",
                "The confirmed translation reached M2M100's target-token limit",
            )

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                **generation_args,
                max_new_tokens=max_new_tokens,
            )

        token_ids = generated[0].tolist()
        if not token_ids or token_ids[-1] != self.tokenizer.eos_token_id:
            raise TranslationProcessingError(
                "target_too_long",
                "M2M100 reached its target-token limit before completing translation",
            )
        decoder_start = self.model.config.decoder_start_token_id
        if token_ids and token_ids[0] == decoder_start:
            token_ids.pop(0)
        if token_ids and token_ids[0] == target_language_id:
            token_ids.pop(0)
        if token_ids and token_ids[-1] == self.tokenizer.eos_token_id:
            token_ids.pop()

        if forced_prefix and token_ids[:len(forced_prefix)] != list(forced_prefix):
            raise TranslationProcessingError(
                "inference_failed",
                "M2M100 did not preserve the confirmed translation prefix",
            )
        return token_ids


class TranslationOnlineProcessor(OnlineProcessorInterface):
    """Decorate online Whisper events with stable incremental translations."""

    def __init__(
        self,
        online,
        translator,
        source_language: str,
        target_language: str,
    ):
        self.online = online
        self.translator = translator
        self.configured_source_language = source_language
        self.target_language = target_language
        self.model_name = translator.model_name
        self._reset_turn()

    def _reset_turn(self):
        self.source_text = ""
        self.detected_source_language = None
        self.previous_hypothesis = None
        self.confirmed_tokens = []
        self.identity_emitted_chars = 0
        self.turn_error = None

    def init(self, offset=None):
        if offset is None:
            self.online.init()
        else:
            self.online.init(offset=offset)
        self._reset_turn()

    def insert_audio_chunk(self, audio):
        self.online.insert_audio_chunk(audio)

    def process_iter(self):
        return self._translate_result(self.online.process_iter(), force_final=False)

    def finish(self):
        return self._translate_result(self.online.finish(), force_final=True)

    def _source_language(self):
        if self.configured_source_language != "auto":
            return self.configured_source_language
        return self.detected_source_language

    def _translation_block(
        self,
        *,
        text: str,
        unconfirmed_text: str,
        status: str,
        model: str | None = None,
    ):
        return {
            "text": text,
            "unconfirmed_text": unconfirmed_text,
            "source_language": self._source_language(),
            "target_language": self.target_language,
            "model": model or self.model_name,
            "status": status,
        }

    def _error_block(self):
        error = self.turn_error
        block = self._translation_block(
            text="", unconfirmed_text="", status="ERROR"
        )
        block["error"] = {"code": error.code, "message": error.message}
        return block

    def _set_error(self, error):
        if isinstance(error, TranslationProcessingError):
            self.turn_error = error
        else:
            logger.exception("M2M100 inference failed", exc_info=error)
            self.turn_error = TranslationProcessingError(
                "inference_failed", f"M2M100 inference failed: {error}"
            )
        return self._error_block()

    def _ensure_language(self):
        source_language = self._source_language()
        if source_language is None:
            raise TranslationProcessingError(
                "source_language_unavailable",
                "Whisper did not provide a detected source language",
            )
        if not self.translator.supports_language(source_language):
            raise TranslationProcessingError(
                "unsupported_source_language",
                f"M2M100 does not support source language '{source_language}'",
            )
        return source_language

    def _identity_update(self, status):
        new_text = self.source_text[self.identity_emitted_chars:]
        self.identity_emitted_chars = len(self.source_text)
        return self._translation_block(
            text=new_text,
            unconfirmed_text="",
            status=status,
            model="identity",
        )

    def _partial_translation(self):
        source_language = self._ensure_language()
        if source_language == self.target_language:
            return self._identity_update("INCOMPLETE")

        current = self.translator.translate(
            self.source_text,
            source_language,
            self.target_language,
            forced_prefix=self.confirmed_tokens,
        )
        old_confirmed = list(self.confirmed_tokens)
        if self.previous_hypothesis is None:
            stable = old_confirmed
        else:
            stable = _longest_common_prefix(self.previous_hypothesis, current)
            if stable[:len(old_confirmed)] != old_confirmed:
                raise TranslationProcessingError(
                    "inference_failed",
                    "M2M100 revised an already confirmed translation prefix",
                )

        old_confirmed_text = self.translator.decode(old_confirmed)
        stable_text = self.translator.decode(stable)
        current_text = self.translator.decode(current)
        confirmed_delta = _decoded_suffix(
            stable_text,
            old_confirmed_text,
            self.translator.decode(stable[len(old_confirmed):]),
        )
        unconfirmed_text = _decoded_suffix(
            current_text,
            stable_text,
            self.translator.decode(current[len(stable):]),
        )
        self.confirmed_tokens = stable
        self.previous_hypothesis = current
        return self._translation_block(
            text=confirmed_delta,
            unconfirmed_text=unconfirmed_text,
            status="INCOMPLETE",
        )

    def _complete_translation(self):
        if self.turn_error is not None:
            return self._error_block()
        source_language = self._ensure_language()
        if source_language == self.target_language:
            return self._identity_update("COMPLETE")

        current = self.translator.translate(
            self.source_text,
            source_language,
            self.target_language,
            forced_prefix=self.confirmed_tokens,
        )
        confirmed_text = self.translator.decode(self.confirmed_tokens)
        current_text = self.translator.decode(current)
        remaining = _decoded_suffix(
            current_text,
            confirmed_text,
            self.translator.decode(current[len(self.confirmed_tokens):]),
        )
        self.confirmed_tokens = current
        self.previous_hypothesis = current
        return self._translation_block(
            text=remaining,
            unconfirmed_text="",
            status="COMPLETE",
        )

    def _handle_event(self, event):
        event = dict(event)
        text = event.get("text")
        is_final = event.get("is_final") is True
        if text:
            self.source_text += text
            detected_language = event.get("detected_language")
            if detected_language:
                self.detected_source_language = detected_language

        if self.turn_error is not None:
            if text or is_final:
                event["translation"] = self._error_block()
        elif is_final and self.source_text:
            try:
                event["translation"] = self._complete_translation()
            except Exception as error:
                event["translation"] = self._set_error(error)
        elif text:
            try:
                event["translation"] = self._partial_translation()
            except Exception as error:
                event["translation"] = self._set_error(error)

        if is_final:
            self._reset_turn()
        return event

    def _translate_result(self, result, force_final):
        was_list = isinstance(result, list)
        events = result if was_list else ([result] if result else [])
        translated = []
        saw_final = False
        for event in events:
            saw_final = saw_final or event.get("is_final") is True
            translated.append(self._handle_event(event))

        if force_final and not saw_final and (self.source_text or self.turn_error):
            final_event = {"is_final": True}
            translated.append(self._handle_event(final_event))

        if was_list or len(translated) > 1:
            return translated
        return translated[0] if translated else {}


def add_m2m100_args(parser):
    group = parser.add_argument_group("M2M100 translation arguments")
    group.add_argument(
        "--translation-model",
        default=DEFAULT_M2M100_MODEL,
        help="Hugging Face model ID or local M2M100 model directory.",
    )
    group.add_argument(
        "--translation-target",
        default="en",
        help="M2M100 target language code (default: en).",
    )
    group.add_argument(
        "--translation-device",
        default=None,
        help="Torch device for M2M100; defaults to the Whisper --device.",
    )
    group.add_argument(
        "--translation-dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="M2M100 inference dtype (default: auto).",
    )


def validate_m2m100_server_args(args):
    if not args.vac:
        raise ValueError("the M2M100 server requires --vac")
    if args.task != "transcribe":
        raise ValueError("the M2M100 server requires --task transcribe")
    if args.out_txt:
        raise ValueError("the M2M100 server emits JSONL and does not support --out-txt")
    if args.translation_target not in M2M100_LANGUAGE_CODES:
        raise ValueError(
            f"unsupported M2M100 target language: {args.translation_target}"
        )
    if args.lan != "auto" and args.lan not in M2M100_LANGUAGE_CODES:
        raise ValueError(f"unsupported M2M100 source language: {args.lan}")
    try:
        translation_device = torch.device(args.translation_device or args.device)
    except RuntimeError as error:
        raise ValueError(f"invalid --translation-device: {error}") from error
    if translation_device.type == "cpu" and args.translation_dtype == "float16":
        raise ValueError("--translation-dtype float16 is not supported on CPU")


def build_m2m100_processor(args, online):
    translator = M2M100Translator(
        model_name=args.translation_model,
        device=args.translation_device or args.device,
        dtype=args.translation_dtype,
    )
    return TranslationOnlineProcessor(
        online=online,
        translator=translator,
        source_language=args.lan,
        target_language=args.translation_target,
    )
