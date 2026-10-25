# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DatasetCropper** is a C# (.NET 10, Windows) image batch-processor that turns images into an AI-training-ready dataset: 1024x1024 crops, per-concept folders, grayscale masks, and WD14 tag captions. It has two front ends over one shared engine:

- **`src/DatasetCropper`** — WPF desktop app: images are dropped into the window (or added via file dialog), then queued and processed.
- **`src/DatasetCropper.Cli`** — command-line front end (`DatasetCropper-cli.exe`): same pipeline, driven by arguments.

Both front ends share **`src/DatasetCropper.Core`**, a plain class library (no WPF) that contains the entire pipeline, the ONNX model wrappers, imaging helpers, and the settings/progress models.

## Build & Run

```bash
dotnet build DatasetCropper.sln
dotnet run --project src/DatasetCropper    # WPF app
dotnet run --project src/DatasetCropper.Cli -- img1.png --concepts cat,dog
dotnet test tests/DatasetCropper.Tests
```

Projects:
- `src/DatasetCropper.Core/DatasetCropper.Core.csproj` — classlib, `net10.0`, no RID, no WPF. Packages: `Microsoft.ML.OnnxRuntime.DirectML`, `SixLabors.ImageSharp`.
- `src/DatasetCropper/DatasetCropper.csproj` — WinExe, `net10.0-windows`, WPF, RID `win-x64`. References Core.
- `src/DatasetCropper.Cli/DatasetCropper.Cli.csproj` — Exe, `net10.0-windows`, RID `win-x64`, AssemblyName `DatasetCropper-cli`. References Core.
- `tests/DatasetCropper.Tests/DatasetCropper.Tests.csproj` — xunit, references the WPF project (which pulls in Core transitively).

## Architecture

### `src/DatasetCropper.Core` — the shared engine (namespace `DatasetCropper.Core`; code lives in `DatasetCropper.Services` / `DatasetCropper.Ml` / `DatasetCropper.Models` / `DatasetCropper.Imaging`)

- `Services/ProcessingService.cs` — **the pipeline** (see below). Also contains `MlHub`, which loads all three models once (each optional) and is shared across the queue. Public surface: `MlHub.Load(modelsFolder, useGpu)`, `new ProcessingService(settings, ml)`, `Run(files, IProgress<ProgressReport>, Action<string> log)`, `Cancel()`, `LastTask`, `IsRunning`.
- `Services/TopazService.cs` — shells out to the Topaz Photo AI CLI (`tpai.exe`) for optional 4x upscaling.
- `Services/SettingsStore.cs` — persists `AppSettings` as JSON at `%APPDATA%\DatasetCropper\settings.json`; also exposes `LogPath`.
- `Ml/` — ONNX model wrappers (all in-process via Microsoft.ML.OnnxRuntime):
  - `Wd14DatasetCropper.cs` — WD14 (waifu-diffusion v1.4) booru DatasetCropper.
  - `GroundingDinoDetector.cs` — open-vocabulary concept detection.
  - `SamSegmenter.cs` — SAM ViT-B (image encoder + mask decoder) for concept masks.
  - `WordPieceTokenizer.cs` — minimal BERT-style tokenizer reading a Hugging Face `tokenizer.json` (used by Grounding DINO).
  - `OnnxFactory.cs` — session creation. Tries the CUDA EP then the DirectML EP (Windows-only, guarded, non-fatal) when `useGpu` is set, else CPU.
  - `ImagePreprocess.cs` — shared pixel/tensor helpers (RGB bytes, CHW tensors, crop+resize, canvas compositing).
  - `ModelChecker.cs` — session-free check of which model files are present.
  - `ModelDownloader.cs` — downloads missing model files (Hugging Face URLs).
- `Imaging/CropTiler.cs` — tile math: base size, overlapping 1024x1024 tiles, concept-centered crops.
- `Imaging/MaskUtils.cs` — per-crop mask extraction/normalization.
- `Models/` — `AppSettings`, `ConceptBox`, `CropRegion`, `ProgressReport` (with `ImageStatus` enum).

### `src/DatasetCropper` — WPF front end only

- `App.xaml.cs` — global unhandled-exception handler (surfaces worker-thread crashes in a MessageBox).
- `MainWindow.xaml(.cs)` — UI shell: a `TabControl` with a **Process** tab (image queue + log), an **Output** tab (thumbnail grid over the output folder) and a **Settings** tab (concepts, folders, models, thresholds, Topaz, GPU — disabled while a run is active). The bottom bar holds Add images / Clear / **View output** (opens the output folder in Explorer) / Start / Cancel. Handles drag-and-drop of image files.
  - Output tab (`OutputViewModel`): concept **folder filter** ("All" + one entry per top-level folder) with counts; **Restore the whole grid** (`RefreshAll`) undoes triage and resets the filter. Confirmed/moved/deleted images are *triaged*, not removed from disk (`_triaged` set), and the next untriaged image is auto-selected. Delete key → `DeleteSelected`; move → `MoveTo`; confirm → `ConfirmAndAdvance`.
  - Process tab: the log **auto-scrolls** to the bottom as output arrives unless the user has scrolled up (wheel/arrow keys disable `_logFollow`; scrolling back to the bottom re-arms it). Delete key removes the selected queue items; **View output** refreshes the grid.
- `ViewModels/MainViewModel.cs` — orchestrates UI state; loads models on a background thread at Start; bridges `IProgress<ProgressReport>` to the queue rows and log. `OpenOutputFolder` creates the output dir (if needed) and launches `explorer.exe` on it.
- `ViewModels/ImageItemViewModel.cs` — one queue row (status + detail).
- `Services/AppTheme.cs` — dark/light theme detection (WPF + Registry).
- `Converters/InverseBoolConverter.cs` — XAML value converter.

### `src/DatasetCropper.Cli` — console front end

`Program.cs` (top-level): hand-rolled argument parsing, then drives the shared engine exactly like the WPF app.
- Positional args: image files and/or directories (directories scanned recursively for png/jpg/jpeg/bmp/webp/tiff — same filter as the WPF file dialog).
- Options: `-c/--concepts`, `--output`, `--models`, `--wd14-threshold`, `--dino-box-threshold`, `--dino-text-threshold`, `--topaz [path]`, `--no-topaz`, `--gpu`, `--cpu`, `--check-models`, `--download-models`, `--help`, `--version`.
- Seeds `AppSettings` from `SettingsStore.Load()` (the same file the WPF app uses), then overlays any provided flags. **Flags are per-invocation and are NOT saved back** — the WPF app owns the persisted settings file.
- `--check-models` / `--download-models` are modes that run and exit (no image paths allowed).
- Console progress: `Total==0` `ProgressReport`s are stage ticks (one in-place line); `Total>0` are per-file completions (own line). Ctrl+C → `ProcessingService.Cancel()` (cooperative), exit 130.
- Exit codes: 0 = finished (per-image errors are logged, not fatal), 1 = fatal, 2 = usage error, 130 = cancelled.

## The Per-Image Pipeline

`ProcessingService.ProcessOne` (in `src/DatasetCropper.Core`) runs, in order:

1. **Optional Topaz 4x upscale** — if enabled and `tpai.exe` is found. The whole queue is upscaled in **one tpai process** (the CLI accepts any number of input files); produced files are matched back per image by base name, so one failure doesn't sink the batch. Images without an output keep their original. Intermediate files land in `<OutputFolder>/.topaz/`. Time cap scales with batch size (30 min per image, 24 h hard max); timeout/crash falls back all images to their originals.
2. **Base image** — resize so the *smallest* dimension is 1024 (aspect preserved). This "base" pixel grid is the coordinate space for detection, tiling, and masks. (Concept-centered crops are an exception — see step 6.)
3. **Concept detection** — Grounding DINO over the base image, prompted with the user's concept list. Boxes come back in base-image pixel coords. If DINO isn't loaded, all crops go to `unknown`.
4. **Crop list** — overlapping 1024x1024 tiles (512px step = 50% overlap, last tile clamped to the edge, so full coverage with no gaps). Each tile is assigned to the concept whose box overlaps it most (intersection area). Additionally, when 2+ distinct concepts are detected, one extra 1024x1024 crop centered on each concept's best box is emitted (suffix `c_<concept>` instead of the tile index).
5. **SAM encoding** — run *once* per base image; then one 1024x1024 mask (SAM-space) per distinct concept, using box center + 2 corners as positive point prompts.
6. **Emit per crop** — for each crop:
   - **Crop source** — *concept-centered* crops (`c_<concept>`) are cut from the **full-resolution original** image, not the 1024 base: the base-space region is mapped back into original-image space (inverse of the base resize) and cropped at 1:1, then resized to 1024x1024, so the concept keeps all original detail. *Tile* crops come from the 1024 base (their framing is base-native).
   - Caption via WD14 (tags ≥ threshold, comma-joined, sorted by confidence) → `<stem>.txt`
   → `<stem>-masklabel.png` (grayscale, concept = white).
   - Crop PNG → `<stem>.png`
       - All written to `<OutputFolder>/<concept>/` (concept name sanitized; `unknown` when unassigned).

   7. **Subject crops** — when `SubjectConcepts` is set (the 2nd concept list), DINO detects those subjects over the base image and each hit emits a 1:1 square of ≥ 1024 px on a side, cut from the **full-resolution original** and written to its own `<OutputFolder>/<subject>/` folder (stem suffix `s_###`). The square (`CropTiler.SubjectSquare`) is centered on the box but **clamped fully inside the image** — its side is capped at min(srcW, srcH) and the window slides to the nearest edge, so crops never contain black padding. `ImagePreprocess.PaddedCrop` (still used at emit) then reduces to a plain crop; the solid-white no-SAM mask fallback fills the whole canvas.

   8. **Optional dedupe stage** — when `StageDedupe` is on (off by default), every emitted crop (composition *and* subject) is dHashed (64-bit difference hash); a crop within 8 Hamming bits of one already emitted **for the same source image** is written to `<OutputFolder>/duplicates/` instead of its concept folder. dHash logic lives in `Imaging/DuplicateDetector.cs` (`DuplicateDetector.DHash`/`Distance` + `DedupeTracker`); the hook is the `WriteCrop` closure inside `EmitJob`.

   ## Models Folder Layout

Models are loaded from subfolders of the user-configured models folder (default `%APPDATA%\DatasetCropper\models`):

```
models/
  wd14/           model.onnx, selected_tags.csv
  grounding-dino/ model.onnx, tokenizer.json
  sam/            image_encoder.onnx, mask_decoder.onnx
```

Each model is optional — missing files are logged and the corresponding stage is skipped (WD14 → empty captions, DINO → all crops `unknown`, SAM → solid white masks). Both front ends degrade gracefully this way.

## Key Conventions & Gotchas

- **ONNX sessions are not thread-safe.** All model calls are serialized through `ProcessingService._mlLock` (a `SemaphoreSlim(1,1)`). Keep it that way if you add models or parallelize.
- **Preprocessing must mirror each model's reference exactly** — the comments in `Wd14DatasetCropper`/`GroundingDinoDetector`/`SamSegmenter` document the exact normalization (e.g. WD14: BGR, no /255, no normalization; DINO & SAM: /255 + ImageNet mean/std). Don't "clean up" these without verifying against the reference.
- **Model I/O names are matched defensively** (substring match with positional fallback) so minor ONNX export variants still work.
- **Coordinate spaces**: base-image pixels (crops, DINO boxes) vs. SAM's 1024x1024 space (`SamEncoding` holds the scale/pad mapping between them). Masks are produced in SAM-space and mapped back per crop.
- Cancellation is cooperative via `CancellationToken`; a 30-minute hard cap exists on Topaz subprocesses.
- Settings sync direction (WPF): UI → `AppSettings` only happens at Start (`SyncSettingsFromUi`); the constructor loads settings into the UI once.
- **CLI settings**: `DatasetCropper-cli` reads the same `%APPDATA%\DatasetCropper\settings.json` but does **not** write it back — flags are per-run. The WPF app owns the persisted file.
- **Native ONNX DLLs** (`onnxruntime*.dll`) reach the two exes via the `win-x64` RID of the app/CLI projects, *not* via Core (Core is plain `net10.0`, no RID). `onnxruntime.dll` (the DirectML build) flows in transitively from the `Microsoft.ML.OnnxRuntime.DirectML` package referenced by Core. `DirectML.dll` is the one exception: the `Microsoft.AI.DirectML` package ships its copy logic in `build/` (not `buildTransitive/`), so it is only imported for a *direct* reference — both front-end projects therefore reference `Microsoft.AI.DirectML` directly and set `<PlatformTarget>x64</PlatformTarget>` (RID-pinned projects don't set it by default, and the copy condition requires a recognized target).
- **`ProgressReport` contract**: `Total==0` means a stage tick (UI/CLI render it in place); `Total>0` means a per-file completion with `OverallFraction`. `ImageStatus.Queued`/`Skipped` are set directly by the WPF ViewModel, never sent through `ProgressReport`.
