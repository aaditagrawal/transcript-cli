"""Exercise optional adapters without model downloads or accelerator libraries."""

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from transcript_cli.engines import _load_engines, dependencies, get_all_engines
from transcript_cli.engines.base import Task, TranscribeOptions
from transcript_cli.engines.faster_whisper import FasterWhisperEngine
from transcript_cli.engines.insanely_fast_whisper import InsanelyFastWhisperEngine
from transcript_cli.engines.mlx_whisper import MLXWhisperEngine
from transcript_cli.engines.parakeet import ParakeetEngine, _segments, _timestamped_words


@pytest.mark.parametrize(
    "engine", [FasterWhisperEngine, InsanelyFastWhisperEngine, MLXWhisperEngine, ParakeetEngine]
)
def test_optional_libraries_can_be_absent(engine, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(dependencies, "import_module", missing)
    _load_engines()
    assert set(get_all_engines()) == {
        "faster-whisper",
        "insanely-fast-whisper",
        "mlx-whisper",
        "parakeet",
    }
    assert engine().is_available() is False


def test_dependency_contract_rejects_incompatible_install(monkeypatch):
    monkeypatch.setattr(dependencies, "import_module", lambda _: ModuleType("faster_whisper"))
    with pytest.raises(ImportError, match="required entry point"):
        dependencies.faster_whisper()


def test_broken_optional_import_is_not_silently_hidden(monkeypatch):
    def broken(name):
        raise RuntimeError("bad shared library")

    monkeypatch.setattr(dependencies, "import_module", broken)
    with pytest.raises(RuntimeError, match="bad shared library"):
        FasterWhisperEngine().is_available()


def test_faster_adapter_preserves_options_and_word_confidence(monkeypatch):
    word = SimpleNamespace(word="hello", start=1.0, end=1.4, probability=0.9)
    segment = SimpleNamespace(text=" hello ", start=1.0, end=1.4, words=[word])
    info = SimpleNamespace(language="fr", language_probability=0.95, duration=2.0)
    pipeline = Mock()
    pipeline.transcribe.return_value = (iter([segment]), info)
    library = SimpleNamespace(
        WhisperModel=Mock(), BatchedInferencePipeline=Mock(return_value=pipeline)
    )
    monkeypatch.setattr(dependencies, "faster_whisper", lambda: library)
    options = TranscribeOptions(
        device="cpu",
        word_timestamps=True,
        language="fr",
        task=Task.TRANSLATE,
        vad_filter=False,
        batch_size=4,
        beam_size=2,
    )
    result = FasterWhisperEngine().transcribe(Path("audio.wav"), "tiny", options)
    library.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    pipeline.transcribe.assert_called_once_with(
        "audio.wav",
        language="fr",
        task="translate",
        word_timestamps=True,
        vad_filter=False,
        batch_size=4,
        beam_size=2,
    )
    assert result.text == "hello"
    assert result.language == "fr"
    assert result.segments[0].words[0].confidence == 0.9
    assert result.duration == 2.0


def test_insanely_adapter_works_without_flash_attention(monkeypatch):
    pipeline = Mock(
        return_value={"text": "Hello", "chunks": [{"text": "Hello", "timestamp": (None, None)}]}
    )
    transformers = SimpleNamespace(pipeline=Mock(return_value=pipeline))
    monkeypatch.setattr(dependencies, "transformers", lambda: transformers)
    monkeypatch.setattr(
        dependencies, "transformers_cache", lambda: SimpleNamespace(cached_file=Mock())
    )
    monkeypatch.setattr(dependencies, "torch", lambda: SimpleNamespace(float16="float16"))
    monkeypatch.setattr(dependencies, "flash_attention_available", lambda: False)
    result = InsanelyFastWhisperEngine().transcribe(
        Path("audio.wav"), "turbo", TranscribeOptions(word_timestamps=True)
    )
    assert transformers.pipeline.call_args.kwargs["model"] == "openai/whisper-large-v3-turbo"
    assert transformers.pipeline.call_args.kwargs["model_kwargs"] == {}
    assert result.segments[0].words[0].start == result.segments[0].words[0].end == 0


def test_mlx_adapter_preserves_translation_and_word_timestamps(monkeypatch):
    library = SimpleNamespace(
        transcribe=Mock(
            return_value={
                "text": " Hello ",
                "language": "es",
                "segments": [
                    {
                        "text": "Hello",
                        "start": 0.1,
                        "end": 0.7,
                        "words": [{"word": "Hello", "start": 0.1, "end": 0.7, "probability": 0.8}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(dependencies, "mlx_whisper", lambda: library)
    result = MLXWhisperEngine().transcribe(
        Path("audio.wav"),
        "tiny",
        TranscribeOptions(task=Task.TRANSLATE, language="es", word_timestamps=True),
    )
    library.transcribe.assert_called_once_with(
        "audio.wav",
        path_or_hf_repo="mlx-community/whisper-tiny-mlx",
        word_timestamps=True,
        language="es",
        task="translate",
    )
    assert result.text == "Hello"
    assert result.language == "es"
    assert result.segments[0].words[0].confidence == 0.8


def test_parakeet_token_durations_group_subwords_and_punctuation():
    tokens: list[dependencies.TokenTimestamp] = [
        {"token": "Hel", "start": 0.2, "end": 0.4},
        {"token": "lo", "start": 0.4, "end": 0.6},
        {"token": ",", "start": 0.6, "end": 0.6},
        {"token": " world", "start": 0.8, "end": 1.2},
        {"token": "!", "start": 1.2, "end": 1.2},
    ]
    words = _timestamped_words("Hello, world!", tokens)
    assert [(w.word, w.start, w.end) for w in words] == [("Hello,", 0.2, 0.6), ("world!", 0.8, 1.2)]
    assert _segments(words, True)[0].words == words
    assert _segments(words, False)[0].words == []


@pytest.mark.parametrize(
    "text,tokens",
    [
        ("missing", []),
        ("hello", [{"token": "wrong", "start": 0.0, "end": 1.0}]),
        ("hello", [{"token": "hello", "start": -1.0, "end": 1.0}]),
        ("hello", [{"token": "hello", "start": 2.0, "end": 1.0}]),
        ("hello", [{"token": "hello", "start": float("nan"), "end": 1.0}]),
    ],
)
def test_parakeet_never_fabricates_invalid_timestamps(text, tokens):
    with pytest.raises(ValueError):
        _timestamped_words(text, tokens)


def test_parakeet_unicode_empty_and_gap_boundaries():
    assert _timestamped_words("", []) == []
    words = _timestamped_words(
        "नमस्ते दुनिया",
        [
            {"token": "नमस्ते", "start": 0.0, "end": 1.0},
            {"token": " दुनिया", "start": 3.0, "end": 4.0},
        ],
    )
    assert [s.text for s in _segments(words, True)] == ["नमस्ते", "दुनिया"]


def test_parakeet_uses_safetensors_duration_decode_and_caches_by_device(monkeypatch):
    model = Mock(device="cuda", dtype="float32")
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = SimpleNamespace(sequences="tokens", durations="durations")
    inputs = Mock()
    inputs.to.return_value = {}
    processor = Mock(return_value=inputs)
    processor.decode.return_value = (["Hello."], [[{"token": "Hello.", "start": 0.0, "end": 0.5}]])
    library = SimpleNamespace(
        AutoModelForTDT=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
    )
    monkeypatch.setattr(dependencies, "parakeet_transformers", lambda: library)
    monkeypatch.setattr(
        dependencies,
        "librosa",
        lambda: SimpleNamespace(load=Mock(return_value=([0] * 16000, 16000))),
    )
    monkeypatch.setattr(dependencies, "torch", lambda: SimpleNamespace(inference_mode=nullcontext))
    engine = ParakeetEngine()
    options = TranscribeOptions(word_timestamps=True)
    result = engine.transcribe(Path("audio.wav"), engine.supported_models[0], options)
    assert result.text == "Hello."
    assert result.duration == 1
    assert result.segments[0].words[0].end == 0.5
    library.AutoModelForTDT.from_pretrained.assert_called_once_with(
        engine.supported_models[0], use_safetensors=True
    )
    processor.decode.assert_called_once_with(
        "tokens", durations="durations", skip_special_tokens=True
    )
    engine._load_model(engine.supported_models[0], "cuda")
    assert library.AutoModelForTDT.from_pretrained.call_count == 1
    engine._load_model(engine.supported_models[0], "cpu")
    assert library.AutoModelForTDT.from_pretrained.call_count == 2


def test_parakeet_download_fetches_only_runtime_weights_and_config(monkeypatch):
    hub = SimpleNamespace(snapshot_download=Mock())
    monkeypatch.setattr(dependencies, "hub", lambda: hub)
    progress = Mock()
    ParakeetEngine().download_model("nvidia/parakeet-tdt-0.6b-v3", progress)
    hub.snapshot_download.assert_called_once_with(
        repo_id="nvidia/parakeet-tdt-0.6b-v3", allow_patterns=["*.json", "*.safetensors"]
    )
    assert progress.call_args_list[0].args[0] == 0.0
    assert progress.call_args_list[-1].args[0] == 1.0


def test_optional_contract_resolves_lazy_module_exports(monkeypatch):
    factory = SimpleNamespace(from_pretrained=Mock())

    class LazyTransformers(ModuleType):
        def __getattr__(self, name: str):
            if name in {"AutoModelForTDT", "AutoProcessor"}:
                return factory
            raise AttributeError(name)

    module = LazyTransformers("transformers")
    monkeypatch.setattr(dependencies, "import_module", lambda _: module)
    api = dependencies.parakeet_transformers()
    assert api.AutoModelForTDT is factory
    assert api.AutoProcessor is factory
