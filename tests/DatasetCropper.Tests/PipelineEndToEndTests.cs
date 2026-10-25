using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DatasetCropper.Models;
using DatasetCropper.Services;
using Xunit;

namespace DatasetCropper.Tests;

/// <summary>
/// Drives the real <see cref="ProcessingService"/> pipeline end-to-end against a
/// synthetic image, with NO model files present. This exercises the exact code
/// path the WPF UI runs (load → base → detect-skip → tiles → mask-skip →
/// caption-skip → emit) and verifies the pipeline degrades gracefully and writes
/// well-formed outputs.
/// </summary>
public class PipelineEndToEndTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "DatasetCropper_e2e_" + Guid.NewGuid().ToString("N"));
    private string _inputDir = "";
    private string _outputDir = "";
    private string _modelsDir = "";

    public PipelineEndToEndTests()
    {
        _inputDir = Path.Combine(_root, "input");
        _outputDir = Path.Combine(_root, "output");
        _modelsDir = Path.Combine(_root, "models"); // empty: no models
        Directory.CreateDirectory(_inputDir);
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_modelsDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { /* best effort */ }
    }

    /// <summary>
    /// Builds a 2048x1024 image whose base grid is (2048,1024), so the tiler
    /// emits exactly 3 tiles (x = 0, 512, 1024; single y row).
    /// </summary>
    private string MakeInputImage()
    {
        using var img = new Image<Rgba32>(2048, 1024);
        // Diagonal gradient so each tile has distinguishable, non-uniform content.
        img.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < img.Height; y++)
            {
                var row = acc.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    row[x] = new Rgba32((byte)(x % 256), (byte)(y % 256), (byte)((x + y) % 256), 255);
            }
        });
        string path = Path.Combine(_inputDir, "sample.png");
        img.SaveAsPng(path);
        return path;
    }

    [Fact]
    public async Task Pipeline_NoModels_WritesWellFormedOutputs()
    {
        string input = MakeInputImage();

        var settings = new AppSettings
        {
            Concepts = new List<string> { "cat", "dog" },
            OutputFolder = _outputDir,
            ModelsFolder = _modelsDir,
            UseTopaz = false,
            UseGpu = false,
        };

        // No model files exist -> all three models load as null.
        var ml = MlHub.Load(_modelsDir, useGpu: false);
        Assert.Null(ml.Wd14);
        Assert.Null(ml.Detector);
        Assert.Null(ml.MaskGen);
        Assert.Equal(3, ml.LoadErrors.Count);

        var service = new ProcessingService(settings, ml);
        var logs = new List<string>();
        var progress = new Progress<ProgressReport>(_ => { });

        service.Run(new[] { input }, progress, logs.Add);
        await service.LastTask!;

        // The run should have completed the single image.
        Assert.Contains(logs, l => l.Contains("sample.png"));
        Assert.Contains(logs, l => l.Contains("Detector not loaded"));
        Assert.Contains(logs, l => l.Contains("Processing finished."));

        // With no DINO, every crop goes to "unsorted". Base is 2048x1024 -> 3 tiles.
        string unsortedDir = Path.Combine(_outputDir, "unsorted");
        Assert.True(Directory.Exists(unsortedDir));

        var pngs = Directory.GetFiles(unsortedDir, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).ToList();
        var masks = Directory.GetFiles(unsortedDir, "*-masklabel.png").ToList();
        var txts = Directory.GetFiles(unsortedDir, "*.txt").ToList();

        Assert.Equal(3, pngs.Count);
        Assert.Equal(3, masks.Count);
        Assert.Equal(3, txts.Count);

        // Each crop is exactly 1024x1024.
        foreach (var p in pngs)
        {
            using var crop = Image.Load<Rgba32>(p);
            Assert.Equal(1024, crop.Width);
            Assert.Equal(1024, crop.Height);
        }
        foreach (var m in masks)
        {
            using var mask = Image.Load<Rgba32>(m);
            Assert.Equal(1024, mask.Width);
            Assert.Equal(1024, mask.Height);
        }

        // No WD14 -> captions are empty strings.
        Assert.All(txts, t => Assert.Equal("", File.ReadAllText(t).Trim()));

        // No masks from SAM -> solid white (255) masks.
        foreach (var m in masks)
        {
            using var mask = Image.Load<Rgba32>(m);
            var center = mask[512, 512];
            Assert.Equal(255, center.R);
        }

        // Topaz disabled -> no .topaz intermediates should exist.
        Assert.False(Directory.Exists(Path.Combine(_outputDir, ".topaz")));
    }

    [Fact]
    public async Task Pipeline_SubjectConcepts_NoDino_NoSubjectOutput()
    {
        // A non-empty 2nd (subject) list with no DINO present must be a no-op:
        // the feature is off without a detector, so the output is identical to
        // the baseline (3 tiles in "unsorted", no subject folder).
        string input = MakeInputImage();

        var settings = new AppSettings
        {
            Concepts = new List<string> { "cat", "dog" },
            SubjectConcepts = new List<string> { "face", "person" },
            OutputFolder = _outputDir,
            ModelsFolder = _modelsDir,
            UseTopaz = false,
            UseGpu = false,
        };

        var ml = MlHub.Load(_modelsDir, useGpu: false);
        Assert.Null(ml.Detector);

        var service = new ProcessingService(settings, ml);
        var logs = new List<string>();
        var progress = new Progress<ProgressReport>(_ => { });

        service.Run(new[] { input }, progress, logs.Add);
        await service.LastTask!;

        string unsortedDir = Path.Combine(_outputDir, "unsorted");
        var pngs = Directory.GetFiles(unsortedDir, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).ToList();
        Assert.Equal(3, pngs.Count);

        // No subject folder was created (no DINO -> subjects skipped).
        Assert.False(Directory.Exists(Path.Combine(_outputDir, "face")));
        Assert.False(Directory.Exists(Path.Combine(_outputDir, "person")));
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public async Task Pipeline_DisabledMaskOrCaption_DoesNotWriteThatSidecar(bool stageMasks, bool stageCaptions)
    {
        string input = MakeInputImage();
        var settings = new AppSettings
        {
            Concepts = new List<string> { "cat" },
            OutputFolder = _outputDir,
            ModelsFolder = _modelsDir,
            StageMasks = stageMasks,
            StageCaptions = stageCaptions,
            UseTopaz = false,
            UseGpu = false,
        };

        using var ml = MlHub.Load(_modelsDir, useGpu: false);
        var service = new ProcessingService(settings, ml);
        service.Run(new[] { input }, new Progress<ProgressReport>(_ => { }), _ => { });
        await service.LastTask!;

        string folder = Path.Combine(_outputDir, "unsorted");
        var crops = Directory.GetFiles(folder, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).ToList();
        var masks = Directory.GetFiles(folder, "*-masklabel.png", SearchOption.TopDirectoryOnly);
        var captions = Directory.GetFiles(folder, "*.txt", SearchOption.TopDirectoryOnly);

        Assert.Equal(3, crops.Count);
        Assert.Equal(stageMasks ? 3 : 0, masks.Length);
        Assert.Equal(stageCaptions ? 3 : 0, captions.Length);
    }

    [Fact]
    public async Task Pipeline_Cancellation_StopsPromptly()
    {
        string input = MakeInputImage();
        var settings = new AppSettings
        {
            OutputFolder = _outputDir,
            ModelsFolder = _modelsDir,
            UseTopaz = false,
            UseGpu = false,
        };
        var ml = MlHub.Load(_modelsDir, false);
        var service = new ProcessingService(settings, ml);
        var logs = new List<string>();
        var progress = new Progress<ProgressReport>(_ => { });

        service.Run(new[] { input }, progress, message =>
        {
            logs.Add(message);
            if (message.Contains(": base ", StringComparison.Ordinal))
                service.Cancel();
        });
        await service.LastTask!; // must NOT throw (a faulted task would propagate OCE)

        // The run stopped cooperatively and logged the cancellation, not a crash.
        Assert.Contains(logs, l => l.Contains("Processing cancelled."));
    }

    [Fact]
    public async Task Pipeline_SameImageTwice_TimestampsCollidingStems()
    {
        string input = MakeInputImage();
        var settings = new AppSettings
        {
            Concepts = new List<string> { "cat", "dog" },
            OutputFolder = _outputDir,
            ModelsFolder = _modelsDir,
            UseTopaz = false,
            UseGpu = false,
        };
        var ml = MlHub.Load(_modelsDir, useGpu: false);
        var progress = new Progress<ProgressReport>(_ => { });

        // First run: baseline stems (no models -> "unsorted", 3 tiles).
        var service1 = new ProcessingService(settings, ml);
        var logs1 = new List<string>();
        service1.Run(new[] { input }, progress, logs1.Add);
        await service1.LastTask!;

        string unsortedDir = Path.Combine(_outputDir, "unsorted");
        var firstRun = Directory.GetFiles(unsortedDir, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).OrderBy(f => f).ToList();
        Assert.Equal(3, firstRun.Count);

        // Second run into the SAME output folder: stems collide -> timestamped
        // copies are written and the originals are preserved.
        var service2 = new ProcessingService(settings, ml);
        var logs2 = new List<string>();
        service2.Run(new[] { input }, progress, logs2.Add);
        await service2.LastTask!;

        var afterSecond = Directory.GetFiles(unsortedDir, "*.png", SearchOption.TopDirectoryOnly)
            .Where(f => !f.EndsWith("-masklabel.png")).OrderBy(f => f).ToList();

        // All 3 originals still present (not overwritten).
        foreach (var f in firstRun)
            Assert.True(File.Exists(f), $"Original was overwritten: {f}");

        // 3 new timestamped crops -> 6 total.
        Assert.Equal(6, afterSecond.Count);
        var timestamped = afterSecond.Except(firstRun).ToList();
        Assert.Equal(3, timestamped.Count);

        // Each timestamped stem ends with _<14 digits> (yyMMddhhmmssff), and its
        // mask + txt were written with the same stem.
        foreach (var f in timestamped)
        {
            string name = Path.GetFileNameWithoutExtension(f);
            Assert.Matches(@"_\d{14}$", name);
            string stem = name;
            Assert.True(File.Exists(Path.Combine(unsortedDir, stem + "-masklabel.png")), "mask missing for " + stem);
            Assert.True(File.Exists(Path.Combine(unsortedDir, stem + ".txt")), "txt missing for " + stem);
        }
    }
}
