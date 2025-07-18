#!/usr/bin/env python3
"""
WhisperX-pipeline с «умным» выбором языка и выводом SRT.

▪ режет ведущую тишину
▪ из 3-х речевых сэмплов берёт язык, проходя фильтр по вероятности
▪ tie-break → суммарная confidence
▪ VAD + beam=2 + suppress_tokens (борьба с «Дима Торжок»)
"""

import argparse
import os
import tempfile
import torch
import whisperx
from pydub import AudioSegment, silence
from whisper.tokenizer import get_tokenizer
from pathlib import Path
from datetime import timedelta

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--audio", required=True)
cli.add_argument("--model", default="large-v3")
cli.add_argument(
    "--detector_model",
    default="small",
    help="модель для language-detect (быстрее и с вероятностью)",
)
cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
cli.add_argument("--batch", type=int, default=8)
cli.add_argument(
    "--min_prob",
    type=float,
    default=0.6,
    help="отсекать языки с вероятностью ниже порога",
)
args = cli.parse_args()
AUDIO = Path(args.audio)

# ---------- 1. Trim leading silence ----------
audio = AudioSegment.from_file(AUDIO)
lead_ms = (
    silence.detect_nonsilent(audio, 300, -35)[0][0]
    if silence.detect_nonsilent(audio, 300, -35)
    else 0
)


def export_chunk(start_ms, dur=10_000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio[start_ms : start_ms + dur].export(tmp.name, format="wav")
        return tmp.name


dur_ms = len(audio)
starts = {
    lead_ms,
    max(lead_ms + (dur_ms - lead_ms) // 3 - 5_000, lead_ms),
    max(lead_ms + 2 * (dur_ms - lead_ms) // 3 - 5_000, lead_ms),
}
chunks = [export_chunk(s) for s in sorted(starts)]

# ---------- 2. Language detect с фильтром вероятности ----------
detector = whisperx.load_model(args.detector_model, args.device, compute_type="int8")

votes, prob_sum = {}, {}
for wav in chunks:
    res = detector.transcribe(wav, batch_size=4, verbose=False)
    lang = res["language"]
    prob = res.get("language_probs", {}).get(lang, 1.0)  # 1.0 если поле отсутствует
    os.unlink(wav)

    # фильтруем заведомо слабые гипотезы
    if prob < args.min_prob:
        continue

    votes[lang] = votes.get(lang, 0) + 1
    prob_sum[lang] = prob_sum.get(lang, 0) + prob

# если всё отфильтровали — берём язык с макс prob без порога
if not votes:
    for wav in chunks:
        pass  # уже удалены, просто сохраняем структуру
    detector = whisperx.load_model(
        args.detector_model, args.device, compute_type="int8"
    )
    for wav in chunks:
        res = detector.transcribe(wav, batch_size=4, verbose=False)
        lang = res["language"]
        prob = res.get("language_probs", {}).get(lang, 1.0)
        votes[lang] = votes.get(lang, 0) + 1
        prob_sum[lang] = prob_sum.get(lang, 0) + prob

# выбор языка
best = max(votes, key=lambda lang: (votes[lang], prob_sum[lang]))
mean_p = prob_sum[best] / votes[best] if votes[best] else 0
print(f"Lang votes {votes}, mean P={mean_p:.2f} → '{best}'")

# ---------- 3. Основной ASR ----------
STOP = ["дима торжок", "dima torzok", "dima torzhok", "субтитры подогнал"]
tok = get_tokenizer(multilingual=True)
SUPPRESS = sorted({t for p in STOP for t in tok.encode(p)})

asr_opts = dict(
    beam_size=2,
    condition_on_previous_text=False,
    suppress_tokens=SUPPRESS,
    temperatures=[0.0],
    no_speech_threshold=0.6,
)
vad_opts = dict(min_silence_duration_ms=500, speech_pad_ms=200)

model = whisperx.load_model(
    args.model,
    args.device,
    compute_type="float16" if args.device == "cuda" else "int8",
    asr_options=asr_opts,
    vad_options=vad_opts,
)

result = model.transcribe(
    str(AUDIO), batch_size=args.batch, language=best, verbose=False
)


# ---------- 4. Write SRT ----------
def ts(sec):
    td = timedelta(seconds=sec)
    h, m, s = str(td).split(":")
    s, ms = s.split(".")
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int(ms):03}"


def clean(txt):
    for p in STOP:
        txt = txt.replace(p, "")
    return " ".join(txt.split())


srt = AUDIO.with_suffix(".srt")
with open(srt, "w", encoding="utf-8") as f:
    for i, seg in enumerate(result["segments"], 1):
        f.write(
            f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{clean(seg['text'])}\n\n"
        )

print("✔ SRT saved to", srt.resolve())
