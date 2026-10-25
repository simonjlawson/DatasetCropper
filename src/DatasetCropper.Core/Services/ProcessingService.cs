using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DatasetCropper.Imaging;
using DatasetCropper.Ml;
using DatasetCropper.Models;

namespace DatasetCropper.Services;

/// <summary>
/// Holds the loaded ML models (each optional). Created once, shared across the
/// processing queue.
/// </summary>
public sealed class MlHub : IDisposable
{
    public ICaptioner? Captioner { get; private set; }
    // Compatibility alias for callers that used the pre-split caption slot.
    public ICaptioner? Wd14 => Captioner;
    public IDetector? Detector { get; private set; }
    public IMaskGenerator? MaskGen { get; private set; }
    public List<string> LoadErrors { get; } = new();

    public static MlHub Load(string modelsFolder, bool useGpu, Action<string>? log = null, MaskModel maskModel = MaskModel.Sam, DetectorModel detectorModel = DetectorModel.GroundingDino, CaptionModel captionModel = CaptionModel.Wd14)
    {
        var hub = new MlHub();
        // Surface the GPU outcome once, up front, when the user asked for the GPU.
        // Each model's Create call re-attaches its own provider, but the availability
        // outcome is identical, so a single probe tells the whole story.
        if (useGpu)
            log?.Invoke(OnnxFactory.ProbeGpu());
        // Captioning is its own slot so a future caption model can be selected
        // without changing the mask model or its lifecycle.
        try
        {
            switch (captionModel)
            {
                case CaptionModel.Wd14:
                    hub.Captioner = Wd14DatasetCropper.TryLoad(Path.Combine(modelsFolder, "wd14"), useGpu, out var e1);
                    if (e1 is not null) hub.LoadErrors.Add("WD14: " + e1);
                    break;
                case CaptionModel.VitGpt2:
                    hub.Captioner = VitGpt2Captioner.TryLoad(Path.Combine(modelsFolder, "vit-gpt2"), useGpu, out var e2);
                    if (e2 is not null) hub.LoadErrors.Add("ViT-GPT2: " + e2);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(captionModel));
            }
        }
        catch (Exception ex) { hub.LoadErrors.Add(captionModel + ": " + ex.Message); }
        // Detector: load only the one selected in settings.
        try
        {
            string? ed = null;
            switch (detectorModel)
            {
                case DetectorModel.Yolo:
                    hub.Detector = YoloDetector.TryLoad(Path.Combine(modelsFolder, "yolo"), useGpu, out ed);
                    if (ed is not null) hub.LoadErrors.Add("YOLO: " + ed);
                    break;
                default:
                    hub.Detector = GroundingDinoDetector.TryLoad(Path.Combine(modelsFolder, "grounding-dino"), useGpu, out ed);
                    if (ed is not null) hub.LoadErrors.Add("DINO: " + ed);
                    break;
            }
        }
        catch (Exception ex) { hub.LoadErrors.Add(detectorModel.ToString() + ": " + ex.Message); }
        // Mask generator: load only the one selected in settings.
        try
        {
            string? em = null;
            switch (maskModel)
            {
                case MaskModel.ClipSeg:
                    hub.MaskGen = ClipSegSegmenter.TryLoad(Path.Combine(modelsFolder, "clipseg"), useGpu, out em);
                    if (em is not null) hub.LoadErrors.Add("CLIPSeg: " + em);
                    break;
                case MaskModel.RembgHuman:
                    hub.MaskGen = RembgHumanSegmenter.TryLoad(Path.Combine(modelsFolder, "rembg"), useGpu, out em);
                    if (em is not null) hub.LoadErrors.Add("Rembg-human: " + em);
                    break;
                default:
                    hub.MaskGen = SamSegmenter.TryLoad(Path.Combine(modelsFolder, "sam"), useGpu, out em);
                    if (em is not null) hub.LoadErrors.Add("SAM: " + em);
                    break;
            }
        }
        catch (Exception ex) { hub.LoadErrors.Add(maskModel.ToString() + ": " + ex.Message); }
        return hub;
    }

    public void Dispose()
    {
        Captioner?.Dispose();
        Detector?.Dispose();
        MaskGen?.Dispose();
    }
}

/// <summary>
/// Orchestrates the image pipeline on a background task.
///
/// The pipeline is <b>stage-major</b>: each stage runs to completion for the whole
/// queue before the next stage begins (Topaz for all images, then base-image for all,
/// then detection for all, …). This keeps each model in use consecutively for the
/// entire queue — friendlier to GPU memory allocation/free and to batching — instead
/// of interleaving DINO→SAM→WD14 per image. It is a reordering only: for the same
/// inputs the emitted files (folders, names, crops, masks, captions) are identical to
/// the former image-major pipeline.
///
/// The trade-off: each image's full-resolution and base pixels are held on its
/// <see cref="ImageJob"/> from the base stage until its emit stage (the end of the
/// queue), so peak memory scales with queue size. For a local dataset-building tool
/// that is the intended trade for the batching benefit.
/// </summary>
public sealed class ProcessingService
{
    private readonly AppSettings _settings;
    private readonly MlHub _ml;
    private readonly SemaphoreSlim _mlLock = new(1, 1); // ONNX sessions are not thread-safe
    private CancellationTokenSource? _cts;
    private Task? _lastTask;

    /// <summary>Folder for images that matched no concept or subject above the detection thresholds.</summary>
    public const string UnsortedFolder = "unsorted";

    /// <summary>The folder near-duplicate crops are moved to.</summary>
    public const string DuplicatesFolder = "duplicates";

    /// <summary>Hamming distance (of 64 dHash bits) within which two crops count as near-duplicates.</summary>
    public const int DedupeMaxDistance = 8;

    /// <summary>When true, emit the extra concept-centered crop in addition to tiles.</summary>
    public bool StageMultiCrop { get; set; } = true;

    public ProcessingService(AppSettings settings, MlHub ml)
    {
        _settings = settings;
        _ml = ml;
    }

    public Task? LastTask => _lastTask;

    public bool IsRunning => _lastTask is { IsCompleted: false };

    public void Cancel() => _cts?.Cancel();

    /// <summary>
    /// Prepends the configured fixed caption prefix (comma-separated, prefix first),
    /// or a bare prefix when WD14 produced no tags. Empty prefix = caption unchanged.
    /// </summary>
    private string ApplyCaptionPrefix(string caption)
    {
        var prefix = (_settings.CaptionPrefix ?? "").Trim();
        if (prefix.Length == 0) return caption;
        return caption.Length == 0 ? prefix : $"{prefix}, {caption}";
    }

    /// <summary>
    /// Per-image state carried across the stages. Stages fill in their outputs; a
    /// stage that throws for a job marks it <see cref="Failed"/> so every later stage
    /// skips it (the run continues with the other images).
    /// </summary>
    private sealed class ImageJob
    {
        public string File = "";
        public string Name = "";
        public int Index;

        // Stage outputs (filled in as each stage runs).
        public string SourcePath = "";          // 1) Topaz output, or the original
        public Image<Rgba32>? Src;              // 2) full-resolution image (AutoOrient applied)
        public Image<Rgba32>? BaseImg;          // 2) base image (smallest dim -> 1024)
        public int BaseW;                       // 2)
        public int BaseH;                       // 2)
        public List<ConceptBox> Boxes = new();  // 3) DINO boxes in base-image coords
        public List<ConceptBox> SubjectBoxes = new(); // 3) 2nd-list boxes, base-image coords
        public List<ConceptBox> MaskBoxes = new();    // 3) optional independent mask-term boxes
        public string ImageConcept = "";        // 4) dominant concept (folder key)
        public string OutFolder = "";           // 4) sanitized folder name
        public List<(CropRegion Region, ConceptBox? Box)> Crops = new(); // 5)
        public List<(CropRegion Region, ConceptBox Box)> SubjectCrops = new(); // 3)
        public MaskState? MaskState;            // 6)
        public Dictionary<string, Image<Rgba32>>? Masks; // 6) concept -> MaskSize x MaskSize mask

        // Error state.
        public bool Failed;
        public string? Error;
    }

    public void Run(IReadOnlyList<string> files, IProgress<ProgressReport> progress, Action<string> log)
    {
        _cts?.Cancel();
        _cts?.Dispose();
        var cts = new CancellationTokenSource();
        _cts = cts;
        var token = cts.Token;

        _lastTask = Task.Run(() =>
        {
            int total = files.Count;
            int done = 0;
            bool cancelled = false;

            // One job per queued file; the stages below operate on this list in order.
            var jobs = new List<ImageJob>(files.Count);
            int index = 0;
            foreach (var file in files)
                jobs.Add(new ImageJob { File = file, Name = System.IO.Path.GetFileName(file), Index = index++ });

            // Run each stage for the whole queue before moving on (stage-major).
            // A stage that throws for one job marks that job failed and continues;
            // a cancellation stops the run cleanly (see the catch below).
            try
            {
                RunStage_Topaz(jobs, progress, log, token, done);
                RunStage_Base(jobs, progress, log, token, done);
                RunStage_Detect(jobs, progress, log, token, done);
                RunStage_Subjects(jobs, progress, log, token, done);
                RunStage_Concept(jobs, progress, log, token, done);
                RunStage_Crops(jobs, progress, log, token, done);
                RunStage_Masks(jobs, progress, log, token, done);
                RunStage_Emit(jobs, progress, log, token, ref done);
            }
            catch (OperationCanceledException)
            {
                // Cancellation is a normal stop, not an error. Swallowing it keeps the
                // task from faulting — a faulted task would surface as an unhandled
                // exception in the UI and skip the ViewModel's post-run cleanup.
                cancelled = true;
            }
            finally
            {
                // Catch-all: the emit stage releases each job's pixels in its own
                // finally, but a run that stopped before emit (cancellation, or a
                // stage that threw) would otherwise leak every job that already built
                // its base image. ReleasePixels is idempotent, so this is safe to run
                // over the whole list regardless of how the run ended.
                foreach (var job in jobs)
                    ReleasePixels(job);
            }

            // Remove any leftover Topaz intermediates (e.g. from a cancelled run).
            try
            {
                string topazDir = System.IO.Path.Combine(_settings.OutputFolder, ".topaz");
                if (System.IO.Directory.Exists(topazDir))
                    System.IO.Directory.Delete(topazDir, true);
            }
            catch { /* best effort */ }
            log(cancelled ? "Processing cancelled." : "Processing finished.");
        }, token);
    }

    // ---- 1) Optional Topaz 4x upscale ----
    private void RunStage_Topaz(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        // nothing to do; every job keeps SourcePath = original
        if (!_settings.UseTopaz || !_settings.StageTopaz) return;
        string? tpai = TopazService.Resolve(_settings.TopazPath);
        if (tpai is null)
        {
            log($"Topaz not found at '{(_settings.TopazPath is "" ? TopazService.DefaultTpaiPath : _settings.TopazPath)}' — using originals.");
            return;
        }
        string upDir = System.IO.Path.Combine(_settings.OutputFolder, ".topaz");
        log("--- Upscaling ---");
        var pending = jobs.Where(j => !j.Failed).ToList();
        if (pending.Count > 0)
        {
            token.ThrowIfCancellationRequested();
            Report(progress, pending[0].File, "Upscaling", pending[0].Index);

            // One tpai process for the whole batch (the CLI accepts any number of
            // input files). Per-image outputs are matched back by base name; any
            // image without an output keeps its original as the pipeline source.
            var results = TopazService.Upscale4xBatch(
                tpai, pending.Select(j => j.File).ToList(), upDir, _settings.TopazArgs,
                new Progress<string>(s => log("  " + s)), token);
            if (results is null)
                log("  Topaz produced no usable output — using originals.");
            foreach (var job in pending)
            {
                string baseName = System.IO.Path.GetFileNameWithoutExtension(job.File).ToLowerInvariant();
                if (results is not null && results.TryGetValue(baseName, out string? upscaled)
                    && !string.IsNullOrEmpty(upscaled))
                {
                    job.SourcePath = upscaled;
                }
                else
                {
                    job.SourcePath = job.File;
                    log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: Topaz produced no output — using original.");
                }
            }
        }
    }

    // ---- 2) Load + build base image (smallest dim -> 1024) ----
    private void RunStage_Base(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                // Default (Topaz disabled or no output): read the original file.
                job.SourcePath = job.SourcePath.Length == 0 ? job.File : job.SourcePath;
                var src = Image.Load<Rgba32>(job.SourcePath);
                // Track the pixels on the job immediately so a failure below (or a
                // later stage) can release them; disposed in RunStage_Emit's finally.
                job.Src = src;
                // Apply EXIF orientation (phone photos are commonly rotated in the
                // sensor frame); everything downstream assumes display orientation.
                src.Mutate(m => m.AutoOrient());
                // The Topaz intermediate is only needed to build the base image —
                // delete it so .topaz/ doesn't grow without bound across runs.
                if (job.SourcePath != job.File)
                {
                    try { System.IO.File.Delete(job.SourcePath); } catch { /* best effort */ }
                }
                var (baseW, baseH) = CropTiler.BaseSize(src.Width, src.Height);
                var baseImg = ImagePreprocess.Resize(src, baseW, baseH);
                job.BaseImg = baseImg;
                job.BaseW = baseW;
                job.BaseH = baseH;
                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: base {baseW}x{baseH}");
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    // ---- 3) Detect concepts (Grounding DINO) ----
    private void RunStage_Detect(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        if (!_settings.StageDetect)
        {
            log("Concept detection disabled — crops go to 'unsorted'.");
            return;
        }
        if (_ml.Detector is null)
        {
            log("Detector not loaded — skipping concept detection (crops go to 'unsorted'; subject concepts skipped).");
            return;
        }
        log("--- Detecting ---");
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                Report(progress, job.File, "Detecting", job.Index);
                _mlLock.Wait(token);
                try { job.Boxes = _ml.Detector.Detect(job.BaseImg!, _settings.Concepts, _settings.DinoBoxThreshold, _settings.DinoTextThreshold); }
                finally { _mlLock.Release(); }

                var results = string.Empty;
                foreach (var box in job.Boxes)
                    results += $" {box.Concept}({box.Score.ToString("0.00")})";
                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: Detected {job.Boxes.Count} box(es):{results}");
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    // ---- 3b) Detect 2nd-list (subject) concepts and build their crops ----
    // The subject list is a separate concern from the composition concepts:
    // each match is a thing to zoom into, so it gets its own output folder
    // (named after the subject) and a 1:1 square crop of at least 1024x1024
    // cut from the full-resolution original. Runs right after detection so it
    // reuses the same DINO session while it is warm.
    private void RunStage_Subjects(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        if (!_settings.StageSubjects) return;                    // stage unchecked
        var subjects = _settings.SubjectConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(c => c.Trim())
            .Distinct()
            .ToList();
        if (subjects.Count == 0) return; // feature off
        if (_ml.Detector is null) return;    // no detector: nothing to find

        log("--- Detecting subjects ---");
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                Report(progress, job.File, "Detecting subjects", job.Index);
                List<ConceptBox> detected;
                _mlLock.Wait(token);
                try { detected = _ml.Detector.Detect(job.BaseImg!, subjects, _settings.DinoBoxThreshold, _settings.DinoTextThreshold); }
                finally { _mlLock.Release(); }

                // Keep every detected subject: each one gets its own 1:1 square
                // crop written to a folder named after its own concept (EmitJob).
                job.SubjectBoxes = detected;

                // One crop per detected subject box: a full-res 1:1 square of
                // at least 1024 on a side, centered on the box and clamped fully
                // inside the image (CropTiler.SubjectSquare). The configured
                // minimum controls how much source resolution/context is retained.
                var crops = new List<(CropRegion Region, ConceptBox Box)>();
                foreach (var box in job.SubjectBoxes)
                {
                    var (x, y, size) = CropTiler.SubjectSquare(box, job.BaseW, job.BaseH,
                        job.Src!.Width, job.Src.Height, _settings.MinimumSubjectResolution);
                    crops.Add((new CropRegion
                    {
                        X = x,
                        Y = y,
                        Width = size,
                        Height = size,
                        Concept = box.Concept,
                        IsSubject = true,
                        SourceBox = box
                    }, box));
                }
                job.SubjectCrops = crops;

                var results = string.Empty;
                foreach (var box in job.SubjectBoxes)
                    results += $" {box.Concept}({box.Score.ToString("0.00")})";
                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: {crops.Count} subject crop(s):{results}");
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    // ---- 4) Determine the dominant concept for each image ----
    private void RunStage_Concept(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                // When multiple chosen concepts match, only the strongest (highest-score)
                // match decides the output folder. All crops for this image are written
                // under that single folder.
                ConceptBox? strongest = job.Boxes.Count > 0
                    ? job.Boxes.OrderByDescending(b => b.Score).First()
                    : null;
                job.ImageConcept = strongest?.Concept ?? UnsortedFolder;
                job.OutFolder = SanitizeFolder(job.ImageConcept);
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    // ---- 5) Build crop list for each image ----
    private void RunStage_Crops(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                Report(progress, job.File, "Cropping", job.Index);
                var crops = new List<(CropRegion Region, ConceptBox? Box)>();
                foreach (var tile in CropTiler.Tile(job.BaseW, job.BaseH, _settings.MinTileOverlap))
                    crops.Add((tile, BestBoxFor(tile, job.Boxes)));

                // One concept-centered crop for the dominant concept, only when multiple
                // distinct concepts matched (a single match is already the folder's focus).
                var distinctConcepts = job.Boxes.Select(b => b.Concept).Distinct().ToList();
                var strongest = job.Boxes.Count > 0 ? job.Boxes.OrderByDescending(b => b.Score).First() : null;
                // Optionally emit one extra concept-centered crop for the dominant concept
                // when multiple distinct concepts were detected. Controlled by the
                // UI-only StageMultiCrop checkbox on the Process tab.
                if (StageMultiCrop && strongest is not null && distinctConcepts.Count >= 2)
                    crops.Add((CropTiler.CenteredOn(strongest, job.BaseW, job.BaseH), strongest));

                if (_settings.StageDetect && _ml.Detector is not null && crops.Count > 1)
                    crops = SelectBestDetectedCrop(job, crops, progress, log, token);

                job.Crops = crops;
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    /// <summary>
    /// Detects each composition crop and keeps only the crop with the strongest
    /// detection for this source image. Exact score ties prefer the crop nearest
    /// the middle of the ordered crop list, keeping the result deterministic.
    /// </summary>
    private List<(CropRegion Region, ConceptBox? Box)> SelectBestDetectedCrop(
        ImageJob job,
        List<(CropRegion Region, ConceptBox? Box)> crops,
        IProgress<ProgressReport> progress,
        Action<string> log,
        CancellationToken token)
    {
        var scores = new float[crops.Count];
        Array.Fill(scores, float.NegativeInfinity);
        bool anyDetection = false;

        for (int i = 0; i < crops.Count; i++)
        {
            token.ThrowIfCancellationRequested();
            Report(progress, job.File, "Selecting crop", job.Index,
                $"crop {i + 1}/{crops.Count}");
            var region = crops[i].Region;
            using var crop = ImagePreprocess.CropAndResize(job.BaseImg!,
                region.X, region.Y, region.Width, region.Height,
                CropTiler.CropSize, CropTiler.CropSize);

            List<ConceptBox> detections;
            _mlLock.Wait(token);
            try
            {
                detections = _ml.Detector!.Detect(crop, _settings.Concepts,
                    _settings.DinoBoxThreshold, _settings.DinoTextThreshold);
            }
            finally { _mlLock.Release(); }

            if (detections.Count > 0)
            {
                scores[i] = detections.Max(d => d.Score);
                anyDetection = true;
            }
        }

        if (!anyDetection) return crops;

        int middle = (crops.Count - 1) / 2;
        int bestIndex = Enumerable.Range(0, crops.Count)
            .Where(i => float.IsFinite(scores[i]))
            .OrderByDescending(i => scores[i])
            .ThenBy(i => Math.Abs(i - middle))
            .ThenBy(i => i)
            .First();
        log($"  [{job.Index + 1}] {job.Name}: selected crop {bestIndex + 1}/{crops.Count} "
            + $"(score {scores[bestIndex]:0.00}); remaining composition crops skipped.");
        return new List<(CropRegion Region, ConceptBox? Box)> { crops[bestIndex] };
    }

    // ---- 6) Mask generation (SAM / CLIPSeg / Rembg-human), once per base image ----
    private void RunStage_Masks(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, int done)
    {
        if (!_settings.StageMasks) { log("Masking disabled — all masks will be solid white."); return; }
        var gen = _ml.MaskGen;
        if (gen is null)
        {
            log("Mask generator not loaded — all masks will be solid white.");
            return;
        }
        log("--- Masking (" + gen.Name + ") ---");
        bool hasMaskTerms = _settings.MaskTerms.Any(c => !string.IsNullOrWhiteSpace(c));
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            // Only images with a dominant concept or subject crops need a SAM mask;
            // the rest fall back to solid white at emit time (same as the former
            // per-image pipeline).
            if (!hasMaskTerms && job.ImageConcept == UnsortedFolder && job.SubjectCrops.Count == 0)
            {
                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: masks solid white (no dominant concept or subjects).");
                continue;
            }
            try
            {
                token.ThrowIfCancellationRequested();
                Report(progress, job.File, "Masking", job.Index);
                _mlLock.Wait(token);
                try
                {
                    var maskTerms = _settings.MaskTerms
                        .Where(c => !string.IsNullOrWhiteSpace(c))
                        .Select(c => c.Trim())
                        .Distinct(StringComparer.OrdinalIgnoreCase)
                        .ToList();
                    if (maskTerms.Count > 0 && _ml.Detector is not null)
                        job.MaskBoxes = _ml.Detector.Detect(job.BaseImg!, maskTerms,
                            _settings.DinoBoxThreshold, _settings.DinoTextThreshold);

                    job.MaskState = gen.Prepare(job.BaseImg!);
                    job.Masks = new Dictionary<string, Image<Rgba32>>();
                    if (maskTerms.Count > 0)
                    {
                        var termMasks = new List<Image<Rgba32>>();
                        foreach (string term in maskTerms)
                        {
                            ConceptBox? box = job.MaskBoxes
                                .Where(b => string.Equals(b.Concept, term, StringComparison.OrdinalIgnoreCase))
                                .OrderByDescending(b => b.Score)
                                .FirstOrDefault();
                            if (gen.Name.Contains("SAM", StringComparison.OrdinalIgnoreCase) && box is null)
                                continue;
                            termMasks.Add(gen.Mask(job.MaskState, term, box));
                        }
                        if (termMasks.Count > 0)
                            job.Masks[MaskTermsKey] = UnionMasks(termMasks);
                        foreach (var mask in termMasks)
                            mask.Dispose();
                    }
                    else if (job.ImageConcept != UnsortedFolder)
                    {
                        var strongest = job.Boxes.OrderByDescending(b => b.Score).First();
                        job.Masks[strongest.Concept] = gen.Mask(job.MaskState, strongest.Concept, strongest);
                    }
                    // One mask per distinct subject concept (DINO NMSes per concept,
                    // so the first box of each is its strongest).
                    foreach (var box in job.SubjectBoxes.GroupBy(b => b.Concept).Select(g => g.First()))
                        job.Masks[box.Concept] = gen.Mask(job.MaskState, box.Concept, box);
                }
                finally { _mlLock.Release(); }

                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: {job.Masks.Count} {gen.Name} mask(s): {string.Join(", ", job.Masks.Keys)}");
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
        }
    }

    // ---- 7) Emit crops ----
    private void RunStage_Emit(List<ImageJob> jobs, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token, ref int done)
    {
        string outRoot = _settings.OutputFolder;
        log("--- Emitting crops ---");
        foreach (var job in jobs)
        {
            if (job.Failed) continue;
            try
            {
                token.ThrowIfCancellationRequested();
                EmitJob(job, outRoot, progress, log, token);
                int totalCrops = job.Crops.Count + job.SubjectCrops.Count;
                string folders = string.Join(", ",
                    (new[] { job.OutFolder }
                        .Concat(job.SubjectCrops.Select(s => SanitizeFolder(s.Box.Concept))))
                    .Distinct());
                log($"  [{job.Index + 1}/{jobs.Count}] {job.Name}: {totalCrops} crop(s) -> {folders}/");
                done++;
                progress.Report(new ProgressReport(jobs.Count, done, job.File, "Done", done / (double)jobs.Count, null));
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex) { Fail(job, ex, log, progress, jobs.Count, done); }
            finally
            {
                // Release this image's pixels now that it has been emitted (or failed).
                // The stage-major design holds them for the queue's duration so a model
                // can be applied to all images consecutively; this is the intended
                // trade-off for a local dataset-building tool.
                ReleasePixels(job);
            }
        }
    }

    private void EmitJob(ImageJob job, string outRoot, IProgress<ProgressReport> progress, Action<string> log, CancellationToken token)
    {
        string srcBase = System.IO.Path.GetFileNameWithoutExtension(job.File);
        var src = job.Src!;
        var baseImg = job.BaseImg!;
        int baseW = job.BaseW, baseH = job.BaseH;
        string imageConcept = job.ImageConcept;
        string outFolder = job.OutFolder;
        var crops = job.Crops;
        var subjectCrops = job.SubjectCrops;
        var masks = job.Masks;
        var maskState = job.MaskState;

        // Dedupe stage (optional): compare this image's crops against each other and
        // send near-duplicates to the shared duplicates folder instead of the concept folder.
        var dedupe = _settings.StageDedupe ? new DedupeTracker(DedupeMaxDistance) : null;
        string duplicatesFolder = System.IO.Path.Combine(outRoot, DuplicatesFolder);

        // Shared write path for both crop kinds. The image is always written;
        // mask and caption sidecars follow their independent stage settings.
        void WriteCrop(string folder, string stem, Image<Rgba32> cropImg,
            Image<Rgba32>? mask, string caption)
        {
            if (dedupe is not null)
            {
                ulong h = DuplicateDetector.DHash(cropImg);
                if (dedupe.IsDuplicate(h, stem, out string? dupOf))
                {
                    log($"    {stem}.png is a near-duplicate of '{dupOf}' - moved to {DuplicatesFolder}/");
                    folder = duplicatesFolder;
                    System.IO.Directory.CreateDirectory(folder);
                }
                else
                {
                    dedupe.Remember(h, stem);
                }
            }

            // Preserve the existing non-destructive behavior unless explicitly
            // told to reuse the stem. The timestamp applies to the whole triplet.
            if (!_settings.OverwriteOutputs
                && System.IO.File.Exists(System.IO.Path.Combine(folder, stem + ".png")))
                stem += "_" + DateTime.Now.ToString("yyMMddhhmmssff");

            var pngRgb = new PngEncoder { ColorType = PngColorType.Rgb, BitDepth = PngBitDepth.Bit8 };
            cropImg.SaveAsPng(System.IO.Path.Combine(folder, stem + ".png"), pngRgb);
            if (_settings.StageMasks && mask is not null)
            {
                var pngGray = new PngEncoder { ColorType = PngColorType.Grayscale };
                mask.SaveAsPng(System.IO.Path.Combine(folder, stem + "-masklabel.png"), pngGray);
            }
            if (_settings.StageCaptions)
            {
                string captionPath = System.IO.Path.Combine(folder, stem + ".txt");
                string generatedCaption = ApplyCaptionPrefix(caption);
                if (_settings.AppendCaptions && System.IO.File.Exists(captionPath))
                {
                    string existing = System.IO.File.ReadAllText(captionPath).Trim();
                    generatedCaption = existing.Length == 0 ? generatedCaption
                        : generatedCaption.Length == 0 ? existing
                        : existing + Environment.NewLine + generatedCaption;
                }
                System.IO.File.WriteAllText(captionPath, generatedCaption);
            }
        }

        // ---- Composition crops (1st concept list) ----
        // Folder is fixed by the strongest match for the whole image so
        // multiple matched concepts don't split outputs across folders.
        int cropIdx = 0;
        int cropTotal = crops.Count;
        string folder = System.IO.Path.Combine(outRoot, outFolder);
        foreach (var (region, box) in crops)
        {
            token.ThrowIfCancellationRequested();
            string concept = imageConcept;
            System.IO.Directory.CreateDirectory(folder);

            // Concept-centered crops are sourced from the full-resolution image
            // (not the 1024 base) so the training crop keeps all the original
            // detail. The region is defined in base-image space; map it back into
            // original-image space (the inverse of the base resize) and crop from
            // there. Tiles still come from the base (their framing is base-native).
            Image<Rgba32> cropSource;
            int cx, cy, cw, ch;
            if (region.IsConceptCentered)
            {
                cropSource = src;
                cx = (int)Math.Round(region.X * (double)src.Width / baseW);
                cy = (int)Math.Round(region.Y * (double)src.Height / baseH);
                cw = (int)Math.Round(region.Width * (double)src.Width / baseW);
                ch = (int)Math.Round(region.Height * (double)src.Height / baseH);
            }
            else
            {
                cropSource = baseImg;
                (cx, cy, cw, ch) = (region.X, region.Y, region.Width, region.Height);
            }
            using var cropImg = ImagePreprocess.CropAndResize(cropSource, cx, cy, cw, ch,
                CropTiler.CropSize, CropTiler.CropSize);

            string suffix = region.IsConceptCentered ? $"c_{outFolder}" : cropIdx.ToString("000");
            string stem = $"{srcBase}_{suffix}";

            // Caption (WD14) — report per-crop progress so a file with many
            // crops doesn't look stuck on a single stage name.
            string caption = "";
            if (_ml.Captioner is not null && _settings.StageCaptions)
            {
                Report(progress, job.File, "Tagging", job.Index, $"crop {cropIdx + 1}/{cropTotal}");
                _mlLock.Wait(token);
                try { caption = _ml.Captioner.Caption(cropImg, _settings.Wd14Threshold); }
                finally { _mlLock.Release(); }
            }

            // Mask (region is base-space for both tile and concept-centered crops).
            using Image<Rgba32>? mask = _settings.StageMasks
                ? MakeMask(masks, maskState, region, box, concept)
                : null;

            WriteCrop(folder, stem, cropImg, mask, caption);
            log($"    {outFolder}/{stem}.png"
                + (_settings.StageMasks ? " (+ mask)" : "")
                + (_settings.StageCaptions
                    ? $" ({(string.IsNullOrWhiteSpace(caption) ? 0 : caption.Split(',').Length)} tag(s))"
                    : ""));
            cropIdx++;
        }

        // ---- Subject crops (2nd concept list) ----
        // Each is a full-res 1:1 square of >= 1024 on a side, cut from the
        // original image, written into a folder named after its own concept.
        int subjectIdx = 0;
        int subjectTotal = subjectCrops.Count;
        foreach (var (region, box) in subjectCrops)
        {
            token.ThrowIfCancellationRequested();
            string concept = box.Concept;
            string subjectFolder = System.IO.Path.Combine(outRoot, SanitizeFolder(concept));
            System.IO.Directory.CreateDirectory(subjectFolder);

            // Region is in full-resolution source space (see RunStage_Subjects /
            // CropTiler.SubjectSquare) and is guaranteed to lie fully inside the
            // image — the square window slides to the image edge instead of
            // padding with black space, so no aspect distortion occurs.
            using var cropImg = ImagePreprocess.PaddedCrop(src,
                region.X, region.Y, region.Width, region.Height,
                CropTiler.CropSize, CropTiler.CropSize, (0, 0, 0));

            string stem = $"{srcBase}_s_{subjectIdx.ToString("000")}";

            string caption = "";
            if (_ml.Captioner is not null && _settings.StageCaptions)
            {
                Report(progress, job.File, "Tagging", job.Index, $"subject {subjectIdx + 1}/{subjectTotal}");
                _mlLock.Wait(token);
                try { caption = _ml.Captioner.Caption(cropImg, _settings.Wd14Threshold); }
                finally { _mlLock.Release(); }
            }

            // Mask: the region is in source space, but SAM masks live in base
            // space, so map the square back to base space before extracting. The
            // padded variant is a no-op guard here since the region is always
            // fully in-bounds.
            var baseRegion = new CropRegion
            {
                X = (int)Math.Round(region.X * (double)baseW / src.Width),
                Y = (int)Math.Round(region.Y * (double)baseH / src.Height),
                Width = (int)Math.Round(region.Width * (double)baseW / src.Width),
                Height = (int)Math.Round(region.Height * (double)baseH / src.Height),
                Concept = concept
            };
            using Image<Rgba32>? mask = _settings.StageMasks
                ? MakeMask(masks, maskState, baseRegion, box, concept)
                : null;

            WriteCrop(subjectFolder, stem, cropImg, mask, caption);
            log($"    {SanitizeFolder(concept)}/{stem}.png"
                + (_settings.StageMasks ? " (+ mask)" : "")
                + (_settings.StageCaptions
                    ? $" ({(string.IsNullOrWhiteSpace(caption) ? 0 : caption.Split(',').Length)} tag(s))"
                    : ""));
            subjectIdx++;
        }
    }

    /// <summary>
    /// Marks a job as failed, releases its pixels, logs the error, and reports an
    /// "Error" completion so the front end can surface it — without aborting the rest
    /// of the queue. <paramref name="done"/> is the number of jobs that have already
    /// finished emit, reported so the overall progress bar isn't reset.
    /// </summary>
    private void Fail(ImageJob job, Exception ex, Action<string> log, IProgress<ProgressReport> progress, int total, int done)
    {
        job.Failed = true;
        job.Error = ex.Message;
        ReleasePixels(job);
        log($"  [{job.Index + 1}/{total}] {job.Name}: ERROR: {ex.Message}");
        progress.Report(new ProgressReport(total, done, job.File, "Error", done / (double)total, ex.Message));
    }

    /// <summary>Releases a job's held pixels (idempotent — safe to call on failure and at emit).</summary>
    private static void ReleasePixels(ImageJob job)
    {
        job.Src?.Dispose();
        job.Src = null;
        job.BaseImg?.Dispose();
        job.BaseImg = null;
    }

    /// <summary>
    /// Mask extraction for both tile crops and subject crops: map the crop
    /// region into the generator's canvas space and resize the slice back to the
    /// 1024 square. The anisotropic stretch is the exact inverse of the
    /// base→canvas mapping (stretch for rembg/CLIPSeg, uniform scale + pad for
    /// SAM), so the mask stays aligned to whatever pixels the crop contains.
    /// The mask canvas always covers the full frame, so the slice is always
    /// inside it (±1px rounding is clamped) — black-canvas padding is NOT the
    /// inverse and would squash the content into a corner with black bands on
    /// the far edges.
    /// </summary>
    private Image<Rgba32> MakeMask(Dictionary<string, Image<Rgba32>>? masks, MaskState? state,
        CropRegion region, ConceptBox? box, string concept)
    {
        Image<Rgba32>? source = null;
        if (masks is not null && state is not null && masks.TryGetValue(MaskTermsKey, out var termMask))
            source = termMask;
        else if (masks is not null && state is not null && box is not null && masks.TryGetValue(concept, out var m))
            source = m;

        if (source is null)
            return MaskUtils.SolidMask(CropTiler.CropSize, 255);

        var (sx, sy, sw, sh) = state.MaskRect(region.X, region.Y, region.Width, region.Height);
        sw = Math.Max(1, sw);
        sh = Math.Max(1, sh);
        // Catmull-Rom (not the default nearest-neighbor) so the canvas -> 1024
        // upscale interpolates smoothly; a light Gaussian blur then breaks up any
        // residual blockiness at the boundary. Both apply only to the mask path —
        // the solid-white fallback above is intentionally left untouched.
        using var cropped = MaskUtils.CropMask(source, sx, sy, sw, sh, CropTiler.CropSize,
            KnownResamplers.CatmullRom);
        using var gray = MaskUtils.ToGrayscaleMask(cropped);
        using var smoothed = MaskUtils.SmoothMask(gray);
        return MaskUtils.ToBinaryMask(smoothed);
    }

    private const string MaskTermsKey = "__mask_terms__";

    private static Image<Rgba32> UnionMasks(IEnumerable<Image<Rgba32>> masks)
    {
        var list = masks.ToList();
        var pixels = list.Select(mask =>
        {
            var values = new byte[mask.Width * mask.Height];
            int index = 0;
            mask.ProcessPixelRows(src =>
            {
                for (int y = 0; y < mask.Height; y++)
                {
                    var row = src.GetRowSpan(y);
                    for (int x = 0; x < row.Length; x++)
                        values[index++] = row[x].R;
                }
            });
            return values;
        }).ToList();
        var result = new Image<Rgba32>(list[0].Width, list[0].Height);
        result.ProcessPixelRows(dst =>
        {
            int index = 0;
            for (int y = 0; y < result.Height; y++)
            {
                var row = dst.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    byte value = 0;
                    foreach (var maskPixels in pixels)
                        value = Math.Max(value, maskPixels[index]);
                    row[x] = new Rgba32(value, value, value, 255);
                    index++;
                }
            }
        });
        return result;
    }

    private static ConceptBox? BestBoxFor(CropRegion tile, List<ConceptBox> boxes)
    {
        ConceptBox? best = null;
        float bestArea = 0;
        foreach (var b in boxes)
        {
            float ix1 = Math.Max(tile.X, b.X), iy1 = Math.Max(tile.Y, b.Y);
            float ix2 = Math.Min(tile.X + tile.Width, b.X2), iy2 = Math.Min(tile.Y + tile.Height, b.Y2);
            float area = Math.Max(0, ix2 - ix1) * Math.Max(0, iy2 - iy1);
            if (area > bestArea) { bestArea = area; best = b; }
        }
        return best;
    }

    public static string SanitizeFolder(string s)
    {
        char[] invalid = System.IO.Path.GetInvalidFileNameChars();
        var sb = new System.Text.StringBuilder();
        foreach (char c in s) sb.Append(invalid.Contains(c) ? '_' : c);
        string r = sb.ToString().Trim();
        return r.Length == 0 ? UnsortedFolder : r;
    }

    private static void Report(IProgress<ProgressReport> progress, string file, string stage, int index, string? message = null) =>
        progress.Report(new ProgressReport(0, 0, file, stage, 0, message));
}
