"""Streaming ECAPA clustering and Whisper output attribution."""

from dataclasses import dataclass

import numpy as np
import torch

from simulstreaming.whisper.whisper_streaming.base import OnlineProcessorInterface

@dataclass
class WindowDecision:
    start: float
    end: float
    speaker: int
    similarity: float


class OnlineSpeakerClustering:
    """Connection-local cosine clustering with two-window hysteresis."""
    def __init__(self, threshold=0.30, max_speakers=0):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.reset()

    def reset(self):
        self.centroids = []
        self.counts = []
        self.current = None
        self.pending = []
        self.pending_target = None

    @staticmethod
    def _normalize(x):
        return x / x.norm(p=2).clamp_min(1e-12)

    def _update(self, speaker, embedding):
        count = self.counts[speaker]
        self.centroids[speaker] = self._normalize(
            self.centroids[speaker] * count + embedding
        )
        self.counts[speaker] = count + 1

    def _candidate(self, embedding):
        if not self.centroids:
            return "new", 1.0
        similarities = torch.stack([torch.dot(embedding, c) for c in self.centroids])
        best = int(torch.argmax(similarities).item())
        similarity = float(similarities[best].item())
        if similarity < self.threshold and (
            self.max_speakers == 0 or len(self.centroids) < self.max_speakers
        ):
            return "new", similarity
        return best, similarity

    def add(self, start, end, embedding):
        embedding = self._normalize(embedding.detach())
        candidate, similarity = self._candidate(embedding)
        record = (start, end, embedding, similarity)
        if self.current is None:
            self.centroids.append(embedding)
            self.counts.append(1)
            self.current = 0
            return [WindowDecision(start, end, 0, 1.0)]

        if candidate == self.current:
            ready = self._flush_pending(self.current)
            self.pending_target = None
            self._update(self.current, embedding)
            ready.append(WindowDecision(start, end, self.current, similarity))
            return ready

        if candidate != self.pending_target:
            ready = self._flush_pending(self.current)
            self.pending = [record]
            self.pending_target = candidate
            return ready

        self.pending.append(record)
        if len(self.pending) < 2:
            return []

        if candidate == "new":
            combined = self._normalize(sum((item[2] for item in self.pending), torch.zeros_like(embedding)))
            self.centroids.append(combined)
            self.counts.append(len(self.pending))
            self.current = len(self.centroids) - 1
            ready = [
                WindowDecision(item[0], item[1], self.current, float(torch.dot(item[2], combined).item()))
                for item in self.pending
            ]
        else:
            self.current = candidate
            for item in self.pending:
                self._update(self.current, item[2])
            ready = [WindowDecision(item[0], item[1], self.current, item[3]) for item in self.pending]
        self.pending = []
        self.pending_target = None
        return ready

    def _flush_pending(self, speaker):
        ready = [WindowDecision(item[0], item[1], speaker, item[3]) for item in self.pending]
        self.pending = []
        return ready

    def flush(self):
        if self.current is None:
            return []
        ready = self._flush_pending(self.current)
        self.pending_target = None
        return ready


class DiarizationOnlineASRProcessor(OnlineProcessorInterface):
    """Runs ECAPA before Whisper and releases only speaker-stable events."""
    def __init__(self, online, encoder, window=2.0, hop=0.5, threshold=0.30, max_speakers=0):
        self.online = online
        self.encoder = encoder
        self.window_samples = int(window * self.SAMPLING_RATE)
        self.hop_samples = int(hop * self.SAMPLING_RATE)
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("speaker window and hop must be positive")
        self.clustering = OnlineSpeakerClustering(threshold, max_speakers)
        self.timeline = []
        self.pending_events = []
        self.init()

    def init(self, offset=None):
        if offset is None:
            self.clustering.reset()
            self.timeline = []
            self.pending_events = []
            offset = 0.0
        self.turn_offset = float(offset)
        self.turn_audio = np.empty(0, dtype=np.float32)
        self.next_window = 0
        self.covered_samples = 0
        self.online.init(offset=offset)

    def warmup(self):
        self.encoder.warmup(self.window_samples)

    def insert_audio_chunk(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        self.turn_audio = np.concatenate((self.turn_audio, audio))
        self._embed_complete_windows()
        # The ECAPA calls above finish before Whisper begins on the same default
        # CUDA stream, preventing overlapping activation-memory peaks.
        self.online.insert_audio_chunk(audio)

    def _embed_complete_windows(self):
        while self.next_window + self.window_samples <= len(self.turn_audio):
            start = self.next_window
            self._embed_window(start, self.turn_audio[start:start + self.window_samples])
            self.covered_samples = start + self.window_samples
            self.next_window += self.hop_samples

    @staticmethod
    def _wrap_pad(audio, target):
        if len(audio) == 0:
            return np.zeros(target, dtype=np.float32)
        repeats = (target + len(audio) - 1) // len(audio)
        return np.tile(audio, repeats)[:target].astype(np.float32, copy=False)

    def _embed_window(self, start_sample, audio, duration_samples=None):
        embedding = self.encoder.embed(audio)
        start = self.turn_offset + start_sample / self.SAMPLING_RATE
        duration_samples = len(audio) if duration_samples is None else duration_samples
        end = start + duration_samples / self.SAMPLING_RATE
        self.timeline.extend(self.clustering.add(start, end, embedding))

    @staticmethod
    def _events(result):
        return result if isinstance(result, list) else ([result] if result else [])

    def process_iter(self):
        self.pending_events.extend(self._events(self.online.process_iter()))
        return self._drain(force=False)

    def finish(self):
        if len(self.turn_audio) and self.covered_samples < len(self.turn_audio):
            start = min(self.next_window, len(self.turn_audio))
            tail = self.turn_audio[start:]
            self._embed_window(
                start, self._wrap_pad(tail, self.window_samples), duration_samples=len(tail)
            )
        self.timeline.extend(self.clustering.flush())
        self.pending_events.extend(self._events(self.online.finish()))
        return self._drain(force=True)

    def _stable_until(self):
        if not self.timeline:
            return float("-inf")
        return max((item.start + item.end) / 2 for item in self.timeline)

    def _drain(self, force):
        ready = []
        retained = []
        stable_until = self._stable_until()
        for event in self.pending_events:
            words = event.get("words", [])
            event_end = max((word["end"] for word in words), default=event.get("end", 0))
            if force or (words and event_end <= stable_until):
                ready.append(self._attribute(event))
            else:
                retained.append(event)
        self.pending_events = retained
        return ready

    def _label_at(self, timestamp):
        if not self.timeline:
            return 0, 1.0
        containing = [item for item in self.timeline if item.start <= timestamp <= item.end]
        choices = containing or self.timeline
        item = min(choices, key=lambda value: abs((value.start + value.end) / 2 - timestamp))
        return item.speaker, item.similarity

    def _attribute(self, event):
        words = event.get("words", [])
        labels = []
        for word in words:
            speaker, similarity = self._label_at((word["start"] + word["end"]) / 2)
            word["speaker"] = f"SPEAKER_{speaker:02d}"
            labels.append((word["speaker"], similarity))
        segments = []
        for word, (speaker, similarity) in zip(words, labels):
            if segments and segments[-1]["speaker"] == speaker:
                segment = segments[-1]
                segment["end"] = word["end"]
                segment["_similarities"].append(similarity)
            else:
                segments.append({
                    "start": word["start"], "end": word["end"], "speaker": speaker,
                    "_similarities": [similarity],
                })
        for segment in segments:
            values = segment.pop("_similarities")
            segment["similarity"] = sum(values) / len(values)
        unique = {label for label, _ in labels}
        event["speaker"] = next(iter(unique)) if len(unique) == 1 else None
        event["speaker_segments"] = segments
        return event
