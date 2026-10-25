using System.Globalization;
using DatasetCropper.Imaging;
using DatasetCropper.Ml;
using DatasetCropper.Models;
using DatasetCropper.Services;

namespace DatasetCropper.Cli;

/// <summary>
/// Command-line front end for the DatasetCropper image pipeline. Thin wrapper over the
/// shared engine in DatasetCropper.Core: it seeds an <see cref="AppSettings"/> from the
/// same %APPDATA%\DatasetCropper\settings.json the WPF app uses, overlays any command-line
/// flags, then drives <see cref="MlHub"/> + <see cref="ProcessingService"/> exactly
/// the way the WPF app does. Flags are per-invocation and are NOT saved back.
/// </summary>
internal static class Program
{
    private const string Usage = """
        DatasetCropper-cli [options] <path>...

        Batch-processes images into a training-ready dataset (1024 crops, per-concept
        folders, grayscale masks, WD14 captions) using the same engine as the WPF app.

        Arguments:
          <path>...            One or more image files and/or directories. Directories are
                               scanned recursively for png/jpg/jpeg/bmp/webp/tiff.

        Options:
          -c, --concepts <list>      Comma-separated concepts (default: from settings)
              --subjects <list>      Comma-separated 2nd-list (subject) concepts: each match
                                     becomes a 1:1 crop (min 1024x1024) in its own folder
              --output <folder>      Output root (default: from settings, else "output")
              --append-captions      Append generated captions to existing .txt files
              --overwrite-output     Overwrite existing crop images and masks
              --no-overwrite-output  Keep timestamped output names on collisions
              --models <folder>      Models folder (default: from settings)
              --wd14-threshold <n>   WD14 tag threshold, 0..1
              --dino-box-threshold <n>
              --dino-text-threshold <n>
              --topaz [path]         Enable Topaz 4x upscale; optional path to tpai.exe
              --no-topaz             Disable Topaz (overrides settings)
              --gpu                  Prefer CUDA/DirectML execution provider
              --cpu                  Force CPU (default when settings say CPU)
              --mask-model <name>    Mask generator: sam, clipseg, or rembg-human
              --caption-model <name> Caption model: wd14 or vit-gpt2 (natural language)
              --detector <name>      Detector: grounding-dino (default, open-vocab) or yolo (COCO-80)
              --check-models         Print model availability and exit
              --download-models      Download missing models and exit
          -h, --help                 Show this help
              --version              Show version

        Settings are read from the shared settings file (same as the WPF app); any
        flag provided here overrides the stored value for this run only.

        Examples:
          DatasetCropper-cli photo.jpg
          DatasetCropper-cli .\\photos --concepts cat,dog --output out
          DatasetCropper-cli C:\\photos --gpu --wd14-threshold 0.4 --topaz
          DatasetCropper-cli --check-models
          DatasetCropper-cli --download-models --models D:\\models
        """;

    private static int Main(string[] args) => RunAsync(args).GetAwaiter().GetResult();

    private static async Task<int> RunAsync(string[] args)
    {
        // ---- Parse ----
        var paths = new List<string>();
        string? concepts = null, subjects = null, output = null, models = null;
        double? wd14 = null, dinoBox = null, dinoText = null;
        bool? topaz = null;
        string? topazPath = null, topazArgs = null;
        bool gpu = false, cpu = false, checkModels = false, downloadModels = false;
        bool? appendCaptions = null, overwriteOutputs = null;
        string? maskModel = null, captionModel = null, detector = null;

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            switch (a)
            {
                case "-h":
                case "--help":
                    Console.WriteLine(Usage);
                    return 0;
                case "--version":
                    Console.WriteLine("DatasetCropper-cli " + typeof(Program).Assembly.GetName().Version);
                    return 0;
                case "-c":
                case "--concepts":
                    concepts = NeedValue(ref i, args, a); if (concepts is null) return 2; break;
                case "--subjects":
                    subjects = NeedValue(ref i, args, a); if (subjects is null) return 2; break;
                case "--output":
                    output = NeedValue(ref i, args, a); if (output is null) return 2; break;
                case "--models":
                    models = NeedValue(ref i, args, a); if (models is null) return 2; break;
                case "--append-captions":
                    appendCaptions = true; break;
                case "--overwrite-output":
                    overwriteOutputs = true; break;
                case "--no-overwrite-output":
                    overwriteOutputs = false; break;
                case "--wd14-threshold":
                    wd14 = NeedDouble(ref i, args, a); if (wd14 is null) return 2; break;
                case "--dino-box-threshold":
                    dinoBox = NeedDouble(ref i, args, a); if (dinoBox is null) return 2; break;
                case "--dino-text-threshold":
                    dinoText = NeedDouble(ref i, args, a); if (dinoText is null) return 2; break;
                case "--topaz":
                    topaz = true;
                    // Optional value: consume the next arg unless it is itself an option.
                    if (i + 1 < args.Length && !IsOption(args[i + 1]))
                        topazPath = args[++i];
                    break;
                case "--no-topaz":
                    topaz = false;
                    break;
                case "--topaz-args":
                    topazArgs = NeedValue(ref i, args, a); if (topazArgs is null) return 2; break;
                case "--gpu":
                    gpu = true;
                    break;
                case "--cpu":
                    cpu = true;
                    break;
                case "--check-models":
                    checkModels = true;
                    break;
                case "--download-models":
                    downloadModels = true;
                    break;
                case "--mask-model":
                    maskModel = NeedValue(ref i, args, a); if (maskModel is null) return 2; break;
                case "--caption-model":
                    captionModel = NeedValue(ref i, args, a); if (captionModel is null) return 2; break;
                case "--detector":
                    detector = NeedValue(ref i, args, a); if (detector is null) return 2; break;
                default:
                    if (a.StartsWith("--concepts=", StringComparison.Ordinal)) { concepts = a["--concepts=".Length..]; break; }
                    if (a.StartsWith("--subjects=", StringComparison.Ordinal)) { subjects = a["--subjects=".Length..]; break; }
                    if (a.StartsWith("--output=", StringComparison.Ordinal)) { output = a["--output=".Length..]; break; }
                    if (a.StartsWith("--models=", StringComparison.Ordinal)) { models = a["--models=".Length..]; break; }
                    if (a.StartsWith("--wd14-threshold=", StringComparison.Ordinal)) { wd14 = ParseThreshold(a["--wd14-threshold=".Length..]); if (wd14 is null) return 2; break; }
                    if (a.StartsWith("--dino-box-threshold=", StringComparison.Ordinal)) { dinoBox = ParseThreshold(a["--dino-box-threshold=".Length..]); if (dinoBox is null) return 2; break; }
                    if (a.StartsWith("--dino-text-threshold=", StringComparison.Ordinal)) { dinoText = ParseThreshold(a["--dino-text-threshold=".Length..]); if (dinoText is null) return 2; break; }
                    if (a.StartsWith("--topaz=", StringComparison.Ordinal)) { topaz = true; topazPath = a["--topaz=".Length..]; break; }
                    if (a.StartsWith("--topaz-args=", StringComparison.Ordinal)) { topazArgs = a["--topaz-args=".Length..]; break; }
                    if (a.StartsWith("--mask-model=", StringComparison.Ordinal)) { maskModel = a["--mask-model=".Length..]; break; }
                    if (a.StartsWith("--caption-model=", StringComparison.Ordinal)) { captionModel = a["--caption-model=".Length..]; break; }
                    if (a.StartsWith("--detector=", StringComparison.Ordinal)) { detector = a["--detector=".Length..]; break; }
                    if (a.StartsWith('-') && a.Length > 1)
                    {
                        Console.Error.WriteLine($"error: unknown option '{a}' (see --help)");
                        return 2;
                    }
                    paths.Add(a);
                    break;
            }
        }

        if ((checkModels || downloadModels) && paths.Count > 0)
        {
            Console.Error.WriteLine("error: --check-models / --download-models take no image paths");
            return 2;
        }

        // ---- Settings: seed from the shared file, overlay flags (flags win) ----
        var settings = SettingsStore.Load();
        if (concepts is not null)
            settings.Concepts = concepts.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        if (subjects is not null)
            settings.SubjectConcepts = subjects.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        if (output is not null) settings.OutputFolder = output;
        if (models is not null) settings.ModelsFolder = models;
        if (appendCaptions is not null) settings.AppendCaptions = appendCaptions.Value;
        if (overwriteOutputs is not null) settings.OverwriteOutputs = overwriteOutputs.Value;
        if (wd14 is not null) settings.Wd14Threshold = wd14.Value;
        if (dinoBox is not null) settings.DinoBoxThreshold = dinoBox.Value;
        if (dinoText is not null) settings.DinoTextThreshold = dinoText.Value;
        if (topaz is not null)
        {
            settings.UseTopaz = topaz.Value;
            if (topazPath is not null) settings.TopazPath = topazPath;
        }
        if (gpu) settings.UseGpu = true;
        if (cpu) settings.UseGpu = false;
        if (maskModel is not null)
        {
            if (!TryParseMaskModel(maskModel, out var parsedMask))
            {
                Console.Error.WriteLine($"error: unknown --mask-model '{maskModel}' (expected sam, clipseg, or rembg-human)");
                return 2;
            }
            settings.MaskModel = parsedMask;
        }
        if (captionModel is not null)
        {
            if (!TryParseCaptionModel(captionModel, out var parsedCaption))
            {
                Console.Error.WriteLine($"error: unknown --caption-model '{captionModel}' (expected wd14 or vit-gpt2)");
                return 2;
            }
            settings.CaptionModel = parsedCaption;
        }
        if (detector is not null)
        {
            if (!TryParseDetector(detector, out var parsedDet))
            {
                Console.Error.WriteLine($"error: unknown --detector '{detector}' (expected grounding-dino or yolo)");
                return 2;
            }
            settings.DetectorModel = parsedDet;
        }
        // NOTE: intentionally NOT calling SettingsStore.Save() — flags are per-run.

        // ---- Modes ----
        if (checkModels)
        {
            var rows = ModelChecker.Check(settings.ModelsFolder);
            Console.WriteLine($"Models folder: {settings.ModelsFolder}");
            foreach (var (name, status) in rows)
                Console.WriteLine($"  {name,-5} {status}");
            return rows.All(r => r.Status == "ready") ? 0 : 1;
        }

        if (downloadModels)
        {
            Directory.CreateDirectory(settings.ModelsFolder);
            var missing = ModelDownloader.FindMissing(settings.ModelsFolder);
            if (missing.Count == 0)
            {
                Console.WriteLine("All models already present.");
                return 0;
            }
            Console.WriteLine($"Downloading {missing.Count} missing file(s) to {settings.ModelsFolder} ...");
            using var dcts = new CancellationTokenSource();
            int dCtrlC = 0;
            Console.CancelKeyPress += (_, e) =>
            {
                if (++dCtrlC >= 2) return; // second press: let the runtime terminate
                e.Cancel = true;
                dcts.Cancel();
            };
            int barLen = 0;
            Action<string> bar = line =>
            {
                if (line.Length > barLen) barLen = line.Length;
                Console.Write("\r" + line.PadRight(barLen));
                Console.Out.Flush();
            };
            int n = ModelDownloader.Download(settings.ModelsFolder, s => Console.WriteLine(s), bar, dcts.Token);
            Console.WriteLine();
            Console.WriteLine($"Downloaded {n} of {missing.Count} file(s).");
            return n == missing.Count ? 0 : 1;
        }

        // ---- Collect files ----
        if (paths.Count == 0)
        {
            Console.Error.WriteLine("error: no image files or directories given (see --help)");
            return 2;
        }
        var files = new List<string>();
        foreach (var p in paths)
        {
            if (Directory.Exists(p))
                files.AddRange(Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
                    .Where(ImageFileFilter.IsImage)
                    .OrderBy(f => f, StringComparer.OrdinalIgnoreCase));
            else if (File.Exists(p) && ImageFileFilter.IsImage(p))
                files.Add(Path.GetFullPath(p));
            else if (File.Exists(p))
                Console.Error.WriteLine($"skipping non-image file: {p}");
            else
            {
                Console.Error.WriteLine($"error: not found: {p}");
                return 2;
            }
        }
        files = files.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        // Drop any .png that collides with a same-stem image in this batch.
        files = ImageFileFilter.FilterFiles(files);
        if (files.Count == 0)
        {
            Console.Error.WriteLine("error: no supported image files found in the given paths");
            return 2;
        }
        if (settings.Concepts.Count == 0)
        {
            Console.Error.WriteLine("error: no concepts defined (use --concepts)");
            return 2;
        }

        // ---- Run ----
        Console.WriteLine($"Loading models from {settings.ModelsFolder} (gpu={settings.UseGpu})...");
        using var ml = MlHub.Load(settings.ModelsFolder, settings.UseGpu, Console.WriteLine, settings.MaskModel, settings.DetectorModel, settings.CaptionModel);
        foreach (var err in ml.LoadErrors)
            Console.WriteLine("  model load: " + err);

        var service = new ProcessingService(settings, ml);
        var reporter = new ConsoleReporter();

        // Synchronous IProgress so progress and log lines stay in order (both run on
        // the worker thread). ProgressReport contract: Total==0 is a stage tick
        // (overwrite one line in place); Total>0 is a per-file completion (own line).
        var progress = new InlineProgress<ProgressReport>(r =>
        {
            string suffix = r.Message is null ? "" : " " + r.Message;
            if (r.Total > 0)
                reporter.Line($"[{r.Completed}/{r.Total}] {r.CurrentImage} — {r.Stage}{suffix}");
            else
                reporter.Stage($"  {r.CurrentImage}: {r.Stage}{suffix}");
        });

        using var cts = new CancellationTokenSource();
        bool cancelled = false;
        int ctrlC = 0;
        Console.CancelKeyPress += (_, e) =>
        {
            if (++ctrlC >= 2) return; // second press: let the runtime terminate
            e.Cancel = true;
            Console.WriteLine();
            Console.WriteLine("Ctrl+C — cancelling after the current step... (press again to force quit)");
            cancelled = true;
            service.Cancel();
            cts.Cancel();
        };

        Console.WriteLine($"Processing {files.Count} image(s) -> {settings.OutputFolder} (concepts: {string.Join(", ", settings.Concepts)})");
        try
        {
            service.Run(files, progress, s => reporter.Line(s));
            await service.LastTask!;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("fatal: " + ex.Message);
            return 1;
        }

        return cancelled ? 130 : 0;
    }

    private static bool TryParseMaskModel(string text, out MaskModel model)
    {
        model = MaskModel.Sam;
        switch (text.Trim().ToLowerInvariant())
        {
            case "sam": model = MaskModel.Sam; return true;
            case "clipseg": model = MaskModel.ClipSeg; return true;
            case "rembg":
            case "rembg-human": model = MaskModel.RembgHuman; return true;
            default: return false;
        }
    }

    private static bool TryParseCaptionModel(string text, out CaptionModel model)
    {
        model = CaptionModel.Wd14;
        switch (text.Trim().ToLowerInvariant())
        {
            case "wd14": model = CaptionModel.Wd14; return true;
            case "vit-gpt2":
            case "vitgpt2": model = CaptionModel.VitGpt2; return true;
            default: return false;
        }
    }

    private static bool TryParseDetector(string text, out DetectorModel model)
    {
        model = DetectorModel.GroundingDino;
        switch (text.Trim().ToLowerInvariant())
        {
            case "grounding-dino":
            case "dino": model = DetectorModel.GroundingDino; return true;
            case "yolo": model = DetectorModel.Yolo; return true;
            default: return false;
        }
    }

    private static bool IsOption(string s) => s.StartsWith('-') && s.Length > 1;

    private static string? NeedValue(ref int i, string[] args, string opt)
    {
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine($"error: {opt} requires a value");
            return null;
        }
        return args[++i];
    }

    private static double? NeedDouble(ref int i, string[] args, string opt)
    {
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine($"error: {opt} requires a value");
            return null;
        }
        return ParseThreshold(args[++i]);
    }

    private static double? ParseThreshold(string s)
    {
        if (!double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) || v < 0 || v > 1)
        {
            Console.Error.WriteLine($"error: threshold '{s}' must be a number between 0 and 1");
            return null;
        }
        return v;
    }

    /// <summary>
    /// An <see cref="IProgress{T}"/> that invokes its handler synchronously on the
    /// reporting thread (no synchronization-context marshaling), keeping console
    /// output ordered with the pipeline's direct log calls.
    /// </summary>
    private sealed class InlineProgress<T> : IProgress<T>
    {
        private readonly Action<T> _handler;
        public InlineProgress(Action<T> handler) => _handler = handler;
        public void Report(T value) => _handler(value);
    }

    /// <summary>
    /// Keeps a single in-place "stage" line (rewritten with \r) separate from
    /// permanent log/progress lines, which first clear the stage line.
    /// </summary>
    private sealed class ConsoleReporter
    {
        private int _stageLen; // length of the last stage line, 0 = none pending

        public void Stage(string text)
        {
            // Pad past the previous (possibly longer) stage text so stale chars
            // don't linger when a shorter stage follows a longer one.
            string line = text.PadRight(Math.Max(_stageLen, text.Length));
            Console.Write("\r" + line);
            Console.Out.Flush();
            _stageLen = line.Length;
        }

        public void Line(string text)
        {
            if (_stageLen > 0)
            {
                Console.Write("\r" + new string(' ', _stageLen));
                _stageLen = 0;
            }
            Console.WriteLine(text);
        }
    }
}
