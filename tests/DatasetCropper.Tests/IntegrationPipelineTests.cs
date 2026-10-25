using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.Extensions.Configuration;
using DatasetCropper.Ml;
using DatasetCropper.Models;
using DatasetCropper.Services;
using Xunit;
using Xunit.Abstractions;

namespace DatasetCropper.Tests;

/// <summary>
/// OPT-IN integration test. It drives the REAL processing pipeline exactly the way
/// the WPF app does (MainViewModel.Start): load the on-disk ONNX models into an
/// <see cref="MlHub"/>, create a <see cref="ProcessingService"/>, queue one image,
/// run the pipeline, and then judge success vs. failure *from the text log* — the
/// same log a user reads in the UI.
///
/// Unlike <see cref="PipelineEndToEndTests"/>, which intentionally runs with NO
/// models, this test runs WITH the models present so the ground-truth path (concept
/// detection, masking, captioning) is exercised on real ONNX sessions.
///
/// ============================================================================
/// HOW TO RUN (this test is NOT executed by default)
/// ----------------------------------------------------------------------------
/// 1. Make sure the models are on disk. The test looks for them at:
///      a) the folder in env var DatasetCropper_MODELS_DIR, if set, else
///      b) the default location in AppSettings.ModelsFolder
///         (%APPDATA%\DatasetCropper\models\ on Windows).
///    Use the app's UI "Download missing models"? to populate them once, or copy an
///    existing models folder in. Run the test only when the models you care about
///    are present — otherwise it SKIPS (never fails) and tells you what's missing.
///
/// 2. Set the opt-in flag and run just this test:
///      Windows PowerShell:
///        $env:DatasetCropper_RUN_INTEGRATION = "1"
///        dotnet test tests\DatasetCropper.Tests\DatasetCropper.Tests.csproj `
///          --filter "FullyQualifiedName~IntegrationPipelineTests"
///
///      CI (exclude this category from the normal suite):
///        dotnet test --filter "Category!=Integration"
///
/// 3. A PASS means the pipeline completed one real image with no "ERROR:" lines in
///    the log; a FAIL prints the full log so the failure point is visible.
///
/// Verdict logic (all derived from the log, mirroring what a user would check):
///   * FAIL  if any log line contains "ERROR:"
///   * FAIL  if it says "Processing cancelled." instead of "Processing finished."
///   * FAIL  the input image name never appears (image was never processed)
///   * DINO loaded  -> a "[n] Detected ... box(es)" line must be present
///   * DINO missing -> a "DINO not loaded" line must be present
/// ============================================================================
/// </summary>
public class IntegrationPipelineTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "DatasetCropper_integration_" + Guid.NewGuid().ToString("N"));
    private string _inputDir = "";
    private string _outputDir = "";
    private string? _modelsDir = null;

    public IntegrationPipelineTests(ITestOutputHelper output)
    {
        _output = output;
        _inputDir = Path.Combine(_root, "input");
        _outputDir = Path.Combine(_root, "output");
        _modelsDir = Path.Combine(_root, "models");
        Directory.CreateDirectory(_inputDir);
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_modelsDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { /* best effort */ }
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task Pipeline_Integration_RealModels_VerifiedByLog()
    {
        // ---- Opt-in gate: skipped (not failed) when not enabled ----
        Skip.If(!IsOptedIn(), "DatasetCropper_RUN_INTEGRATION not set (integration tests are opt-in)");

        // ---- Resolve a models folder that actually has models, else skip ----
        string? modelsDir = ResolveModelsFolder();
        Skip.If(modelsDir is null, "No models folder with ready models found. Populate one or set DatasetCropper_MODELS_DIR.");
        _output.WriteLine("Using models folder: " + modelsDir);

        // ---- Queue one synthetic image (same shape the app would process) ----
        string input = MakeInputImage();
        _output.WriteLine("Queued input image: " + input);

        var settings = new AppSettings
        {
            Concepts = new List<string> { "cat", "dog" },
            OutputFolder = _outputDir,
            ModelsFolder = modelsDir,
            UseTopaz = false,
            UseGpu = false, // deterministic CPU EP; flip to true to exercise CUDA
        };

        // ---- Execute the project's pipeline exactly as MainViewModel.Start does ----
        var ml = MlHub.Load(modelsDir, settings.UseGpu);
        foreach (var loadErr in ml.LoadErrors) _output.WriteLine("  [model load] " + loadErr);
        _output.WriteLine($"Models loaded: WD14={(ml.Wd14 is null ? "no" : "yes")}, " +
            $"DINO={(ml.Detector is null ? "no" : "yes")}, MASK={(ml.MaskGen is null ? "no" : "yes")}");

        var service = new ProcessingService(settings, ml);
        var log = BuildThreadSafeLog(_output);
        var progress = new Progress<ProgressReport>(_ => { });

        service.Run(new[] { input }, progress, log.Append);
        await service.LastTask!; // must not fault

        string runLog = log.Value;
        _output.WriteLine("----- Captured pipeline log -----");
        _output.WriteLine(runLog);
        _output.WriteLine("----- End of pipeline log -----");

        // ---- Verdict, read straight from the log the user would see ----
        string filename = Path.GetFileName(input);

        var errorLines = log.Lines.Where(l => l.Contains("ERROR:")).ToList();
        if (errorLines.Count > 0)
            Assert.Fail("Pipeline reported ERROR(s) during processing:\n  " + string.Join("\n  ", errorLines)
                + "\n\nFull log:\n" + runLog);

        if (log.Lines.Any(l => l.Contains("Processing cancelled.")) && !log.Lines.Any(l => l.Contains("Processing finished.")))
            Assert.Fail("Pipeline did not finish (cancelled or aborted):\n" + runLog);

        if (!log.Lines.Any(l => l.Contains(filename)))
            Assert.Fail($"Input image '{filename}' was never processed:\n" + runLog);

        if (ml.Detector is not null)
        {
            if (!log.Lines.Any(l => l.Contains("box(es)")))
                Assert.Fail("DINO was loaded but no 'Detected ... box(es)' line appeared:\n" + runLog);
        }
        else
        {
            if (!log.Lines.Any(l => l.Contains("DINO not loaded")))
                Assert.Fail("DINO was not loaded but the 'DINO not loaded' notice is missing:\n" + runLog);
        }

        // Reached here: every success signal from the log is present and no ERRORs.
        Assert.True(log.Lines.Any(l => l.Contains("Processing finished.")),
            "Expected a 'Processing finished.' line in the log:\n" + runLog);
    }

    /// <summary>
    /// Real-image semantic integration test: feeds the committed photo of a man
    /// (tests\DatasetCropper.Tests\sampleImage\dscf0015.jpg) through the full pipeline with
    /// the concepts ["man","face"] and asserts that BOTH keywords are actually
    /// processed — i.e. Grounding DINO reports a detection for each, and the image
    /// is cropped into the dominant (highest-scored) concept's folder per the
    /// dominant-concept design. Skipped (not failed) when the models or the sample
    /// image aren't available.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task Pipeline_Integration_ManAndFace_BothProcessed()
    {
        Skip.If(!IsOptedIn(), "DatasetCropper_RUN_INTEGRATION not set (integration tests are opt-in)");

        // The sample photo is committed to the repo; locate it relative to the
        // test-assembly output directory by walking up to the repo root.
        string? input = FindSampleImage();
        Skip.If(input is null, "sampleImage/dscf0015.jpg not found. Place it under tests/DatasetCropper.Tests/sampleImage/");

        string? modelsDir = ResolveModelsFolder();
        Skip.If(modelsDir is null, "No models folder with ready models found. Populate one or set DatasetCropper_MODELS_DIR.");

        // Concept detection (the "man"/"face" check) is owned by Grounding DINO;
        // without it there is nothing to verify, so skip rather than fail.
        var preLoad = MlHub.Load(modelsDir, useGpu: false);
        if (preLoad.Detector is null)
        {
            preLoad.Dispose();
            Skip.If(true, "Grounding DINO is not loaded (model.onnx / tokenizer.json missing from grounding-dino/).");
        }
        preLoad.Dispose();

        var settings = new AppSettings
        {
            Concepts = new List<string> { "man", "face" },
            OutputFolder = _outputDir,
            ModelsFolder = modelsDir,
            UseTopaz = false,
            UseGpu = false,
        };

        var ml = MlHub.Load(modelsDir, settings.UseGpu);
        foreach (var loadErr in ml.LoadErrors) _output.WriteLine("  [model load] " + loadErr);
        Assert.True(ml.Detector is not null, "Expected Grounding DINO to load from " + modelsDir);

        string filename = Path.GetFileName(input);
        var service = new ProcessingService(settings, ml);
        var log = BuildThreadSafeLog(_output);
        var progress = new Progress<ProgressReport>(_ => { });

        service.Run(new[] { input }, progress, log.Append);
        await service.LastTask!; // must not fault

        string runLog = log.Value;
        var errorLines = log.Lines.Where(l => l.Contains("ERROR:")).ToList();
        if (errorLines.Count > 0)
            Assert.Fail("Pipeline reported ERROR(s):\n  " + string.Join("\n  ", errorLines) + "\n\nFull log:\n" + runLog);
        Assert.True(log.Lines.Any(l => l.Contains(filename)),
            $"Input image '{filename}' was never processed:\n" + runLog);
        Assert.True(log.Lines.Any(l => l.Contains("Processing finished.")),
            "Expected 'Processing finished.' in the log:\n" + runLog);

        // --- Keyword check, read straight from the DINO detection line ---
        //   format: "  Detected {n} box(es): man(0.66) face(0.74) ..."
        string? detectionLine = log.Lines.FirstOrDefault(l => l.Contains("box(es):"));
        Assert.True(detectionLine is not null, "No 'Detected ... box(es)' line found — nothing was detected:\n" + runLog);

        string conceptsPart = detectionLine![(detectionLine!.IndexOf("box(es):", StringComparison.Ordinal) + "box(es):".Length)..].Trim();
        var detected = conceptsPart
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => s.Contains('(') ? s[..s.IndexOf('(')] : s)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        _output.WriteLine("Detected concepts from log: " + string.Join(", ", detected));

        Assert.Contains("man", detected, StringComparer.OrdinalIgnoreCase);
        Assert.Contains("face", detected, StringComparer.OrdinalIgnoreCase);

        // --- Output check: the dominant-concept design routes ALL of this image's
        // crops into the highest-scored concept's single folder ---
        var scored = conceptsPart
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(s => s.Contains('('))
            .Select(s => (Concept: s[..s.IndexOf('(')],
                          Score: float.Parse(s[(s.IndexOf('(') + 1)..s.IndexOf(')')],
                              System.Globalization.CultureInfo.InvariantCulture)))
            .ToList();
        string dominant = scored.OrderByDescending(s => s.Score).First().Concept;
        _output.WriteLine("Dominant concept: " + dominant);
        string folder = Path.Combine(_outputDir, dominant);
        Assert.True(Directory.Exists(folder), $"Expected dominant output folder '{folder}' but it was not created:\n" + runLog);
        var crops = Directory.EnumerateFiles(folder, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).ToList();
        Assert.True(crops.Count > 0, $"Expected at least one crop in '{folder}'; found none:\n" + runLog);
    }

    private static readonly Lazy<IConfiguration> _userSecrets = new(() =>
    {
        var builder = new ConfigurationBuilder();
        string? jsonPath = UserSecretsPath();
        if (!string.IsNullOrEmpty(jsonPath) && File.Exists(jsonPath))
            builder.AddJsonFile(jsonPath, optional: true, reloadOnChange: false);
        return builder.Build();
    });

    private static bool IsOptedIn()
    {
        if (string.Equals(_userSecrets.Value["DatasetCropper_RUN_INTEGRATION"], "1", StringComparison.Ordinal))
            return true;
        return string.Equals(Environment.GetEnvironmentVariable("DatasetCropper_RUN_INTEGRATION"), "1", StringComparison.Ordinal);
    }

    /// <summary>
    /// Resolves the user-secrets JSON file the same way the .NET CLI does:
    /// %DOTNET_CLI_HOME%\user\secrets\{UserSecretsId}\secrets.json if DOTNET_CLI_HOME
    /// is set, otherwise %APPDATA%\Microsoft\UserSecrets\{UserSecretsId}\secrets.json.
    /// </summary>
    private static string? UserSecretsPath()
    {
        const string secretsId = "2fc26c47-8d61-4503-9cd6-d6240885f523";
        string? cliHome = Environment.GetEnvironmentVariable("DOTNET_CLI_HOME");
        string baseDir = string.IsNullOrEmpty(cliHome)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Microsoft", "UserSecrets")
            : Path.Combine(cliHome, "user", "secrets");
        string path = Path.Combine(baseDir, secretsId, "secrets.json");
        return File.Exists(path) ? path : null;
    }

    /// <summary>
    /// Returns the first candidate models folder where at least one expected model
    /// is present, or null if none are ready (so the test skips rather than fails).
    /// </summary>
    private string? ResolveModelsFolder()
    {
        var candidates = new List<string>();
        string? envDir = Environment.GetEnvironmentVariable("DatasetCropper_MODELS_DIR");
        if (!string.IsNullOrWhiteSpace(envDir)) candidates.Add(envDir);
        candidates.Add(new AppSettings().ModelsFolder);
        candidates.Add(SettingsStore.Load().ModelsFolder);

        foreach (var dir in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)) continue;
            var statuses = ModelChecker.Check(dir);
            if (statuses.Any(s => s.Status == "ready"))
            {
                _output.WriteLine("  models at " + dir + " -> " + string.Join("  ", statuses.Select(s => $"{s.Name}:{s.Status}")));
                return dir;
            }
        }
        return null;
    }

    /// <summary>
    /// Builds a 2048x1024 gradient image (base grid 2048x1024 -> 3 tiles), so the
    /// integration run produces a small, deterministic number of crops.
    /// </summary>
    /// <summary>
    /// Locates the committed sample photo. At run time the test assembly lives
    /// under \bin\, so the repo-relative path sampleImage\dscf0015.jpg is found by
    /// walking up from the current directory until the file is hit; this keeps the
    /// test working from any launch directory without hardcoding the drive letter.
    /// </summary>
    private static string? FindSampleImage()
    {
        DirectoryInfo? start = new(Directory.GetCurrentDirectory());
        for (var d = start; d is not null; d = d.Parent)
        {
            // Covers both "...\<solution>\tests\DatasetCropper.Tests\sampleImage\..." and
            // "...\<solution>\sampleImage\..." depending on where the process runs from.
            string candidate = Path.Combine(d.FullName, "DatasetCropper.Tests", "sampleImage", "dscf0015.jpg");
            if (File.Exists(candidate)) return candidate;
            string alt = Path.Combine(d.FullName, "sampleImage", "dscf0015.jpg");
            if (File.Exists(alt)) return alt;
        }
        return null;
    }

    private string MakeInputImage()
    {
        using var img = new Image<Rgba32>(2048, 1024);
        img.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < img.Height; y++)
            {
                var row = acc.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    row[x] = new Rgba32((byte)(x % 256), (byte)(y % 256), (byte)((x + y) % 256), 255);
            }
        });
        string path = Path.Combine(_inputDir, "integration_sample.png");
        img.SaveAsPng(path);
        return path;
    }

    private sealed class LogCollector
    {
        private readonly object _lock = new();
        private readonly List<string> _lines = new();
        private readonly ITestOutputHelper _output;

        public LogCollector(ITestOutputHelper output) { _output = output; }

        public void Append(string line)
        {
            lock (_lock) _lines.Add(line);
            _output.WriteLine(line);
        }

        public List<string> Lines { get { lock (_lock) return _lines.ToList(); } }
        public string Value { get { lock (_lock) return string.Join(Environment.NewLine, _lines); } }
    }

    private static LogCollector BuildThreadSafeLog(ITestOutputHelper output) => new(output);
}
