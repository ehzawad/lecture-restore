# Lecture Restore

Local lecture restoration with FFmpeg and a reviewed original-slide compositor. Media stays on your computer. The repository contains code and tests; recordings, slides, models, recipes and generated results stay in a separate workspace.

The baseline preserves the original input, compares conservative audio/video presets, creates loudness-matched audio and comparison frames, and requires preview acceptance before a full encode. Validation checks duration, frame cadence, aspect ratio, audio/video timing, decoding and audio peaks. These measurements do not certify every word or mathematical symbol.

## Setup

Use Python 3.10+ and FFmpeg/ffprobe with `libx264`. Processing uses CPU on macOS, including Apple Silicon and Intel; CUDA and proprietary software are unnecessary. The pipeline has been exercised with FFmpeg 9.0.2. Different FFmpeg builds can produce different encoded bytes.

From the cloned repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Keep each recording in its own local workspace. `--workspace` goes before the command and defaults to the current directory. Default source, output, reports and logs use that workspace. Explicit path arguments are relative to your current shell directory. Presets and implementation hashes come from the installed code.

For several lectures, keep the code checkout beside a folder of recording workspaces:

```text
videos/
├── lecture-restore/              # This GitHub repository
└── records/
    └── <lecture-name>/
        ├── source/              # Original recording and any original slide PDF
        ├── output/              # This lecture's restored video and previews
        ├── recipe.json          # For an accepted slide restoration
        ├── slides/              # Verified PDF renderings, when used
        └── masks/               # Reviewed cursor masks, when used
```

Point `--workspace` at `records/<lecture-name>`. Each lecture keeps its own outputs and local reports; recipes use relative paths so the recording folder can move as a unit. Keep generated files outside the code checkout.

## Free FFmpeg baseline

Reference an existing local recording without copying it:

```sh
lecture-restore --workspace /path/to/course inspect --input /path/to/lecture.mp4
```

Select representative 30–60-second sections for the recording and save `/path/to/course/segments.json`. Include speech, slide text/equations and lecturer movement. For example, adjust these timestamps to your actual material:

```json
{"segments": [
  {"name": "speech", "start": 60, "duration": 30},
  {"name": "equations", "start": 450, "duration": 30},
  {"name": "movement", "start": 900, "duration": 30}
]}
```

```sh
lecture-restore --workspace /path/to/course previews \
  --input /path/to/lecture.mp4 --segments /path/to/course/segments.json \
  --presets conservative conservative_2x
```

Open the generated `output/previews/index.html`, compare the video and matched audio, then accept the configuration you actually reviewed:

```sh
lecture-restore --workspace /path/to/course accept \
  --preset conservative_2x --audio baseline \
  --reviewed speech,symbols,faces,handwriting,sync,motion \
  --note "Describe the checks you made and any remaining limitations."
lecture-restore --workspace /path/to/course restore
```

The result is `/path/to/course/output/lecture_restored.mp4` with a neighboring validation JSON. Existing outputs are never overwritten. Add `--resume` to a preview invocation only when its source and settings are unchanged. Source audio is retained as a local reference; neither baseline nor neural output replaces the source recording.

Presets include conservative spatial/temporal denoising, conditional deinterlacing, optional deblocking, mild sharpening and 2× scaling. Change settings in your own JSON passed with `--config`; regenerate and review previews after changes. Low-resolution video does not contain unlimited recoverable detail.

## Replay a reviewed slide restoration

`scripts/replay_recipe.py` reproduces an already reviewed local recipe using original PDF renderings plus the recorded lecturer and sparse cursor overlays. Genuine animations and uncertain content retain the source video. It uses deterministic filters rather than generated text or faces.

The currently supported compositor profile is **1280×720 at 30000/1001 fps, source time base 1/30000, to 2560×1440**. It is separate from the more general FFmpeg baseline. Each new course needs its own verified slide timeline, camera rectangles, pointer masks and preview review. This repository does not automatically map arbitrary PDFs or reuse another course's annotations.

A local recipe records asset hashes, exact segment boundaries, settings and acceptance. Its paths resolve relative to the recipe file; source video, PDF renders and masks are not uploaded or bundled with the code. Keep these local assets to reproduce the result.

Verify selected complete original segments against an existing final video:

```sh
python3 scripts/replay_recipe.py \
  --recipe /path/to/records/recipe.json \
  --reference /path/to/final.mp4 --segments 19 50 74 75 \
  --report /path/to/records/reproduction.json --verify-audio
```

Choose segment numbers present in your recipe. Complete original segment boundaries retain the encoder's prediction context; arbitrary trimmed previews need not be byte-identical. The verifier renders to temporary storage, compares decoded frames and removes temporary media. Its JSON records measured results.

To reproduce the entire approved recipe into a new file:

```sh
python3 scripts/replay_recipe.py --recipe /path/to/records/recipe.json \
  --full --output /path/to/new-result.mp4 --report /path/to/records/full-replay.json
```

Use `--work /path/to/work` and `--resume` to recover an interrupted full replay with unchanged inputs. Keep enough free disk space for temporary video chunks and the final MP4. Completed outputs are published only after validation.

## Optional local speech enhancement

DeepFilterNet is optional and is not used by the reviewed slide compositor. Its isolated setup targets native macOS arm64/x86_64 with Python 3.11. It downloads packages and a hash-verified model, never lecture media. Installation or model failure leaves the FFmpeg baseline usable.

```sh
python3 scripts/setup_neural.py --project-root /path/to/course --python /path/to/python3.11
lecture-restore --workspace /path/to/course previews \
  --input /path/to/lecture.mp4 --segments /path/to/course/segments.json --neural
```

The pinned requirements are included for reproducibility; compatible wheels may not exist on every system. Review neural speech alongside the original before selecting `--audio neural`. `--check-only` checks an existing setup without downloading. Neural processing can change speech, and measured format/timing checks cannot establish word fidelity.

There is no YouTube downloader, media upload, paid service integration or automatic quality guarantee. Start from a local recording you can process and retain the final file plus the local recipe/assets needed for future replay.
