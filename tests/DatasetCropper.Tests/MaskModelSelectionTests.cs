using System.IO;
using DatasetCropper.Ml;
using DatasetCropper.Models;
using DatasetCropper.Services;
using Xunit;

namespace DatasetCropper.Tests;

/// <summary>
/// Verifies the mask-model selection feature: choosing CLIPSeg or Rembg-human
/// with those model files absent degrades gracefully (null generator, a recorded
/// load-error note, and no throw) — the same contract the pre-existing SAM path
/// already honors.
/// </summary>
public class MaskModelSelectionTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "DatasetCropper_maskmodel_" + Guid.NewGuid().ToString("N"));

    public MaskModelSelectionTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { /* best effort */ }
    }

    [Theory]
    [InlineData(MaskModel.ClipSeg, "CLIPSeg")]
    [InlineData(MaskModel.RembgHuman, "Rembg-human")]
    public void SelectingMaskModel_WhenFilesMissing_IsGraceful(MaskModel model, string errorPrefix)
    {
        var lines = new List<string>();
        var hub = MlHub.Load(_root, useGpu: false, log: lines.Add, maskModel: model);

        Assert.Null(hub.MaskGen);
        Assert.Contains(hub.LoadErrors, e => e.StartsWith(errorPrefix, StringComparison.Ordinal));

        hub.Dispose();
    }

    [Fact]
    public void ClipSegSegmenter_TryLoad_MissingFolder_ReturnsNull()
        => Assert.Null(ClipSegSegmenter.TryLoad(Path.Combine(_root, "nope"), useGpu: false, out var err));

    [Fact]
    public void ClipSegSegmenter_TryLoad_MissingFolder_LogsError()
    {
        ClipSegSegmenter.TryLoad(Path.Combine(_root, "nope"), useGpu: false, out var err);
        Assert.False(string.IsNullOrWhiteSpace(err));
    }

    [Fact]
    public void RembgHumanSegmenter_TryLoad_MissingFolder_ReturnsNull()
        => Assert.Null(RembgHumanSegmenter.TryLoad(Path.Combine(_root, "nope"), useGpu: false, out var err));

    [Fact]
    public void RembgHumanSegmenter_TryLoad_MissingFolder_LogsError()
    {
        RembgHumanSegmenter.TryLoad(Path.Combine(_root, "nope"), useGpu: false, out var err);
        Assert.False(string.IsNullOrWhiteSpace(err));
    }
}
