using System.IO;
using DatasetCropper.Models;
using DatasetCropper.Services;
using Xunit;

namespace DatasetCropper.Tests;

public sealed class CaptionModelSelectionTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "DatasetCropper_captionmodel_" + Guid.NewGuid().ToString("N"));

    public CaptionModelSelectionTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    [Fact]
    public void SelectingVitGpt2_WhenFilesMissing_IsGraceful()
    {
        using var hub = MlHub.Load(_root, useGpu: false, captionModel: CaptionModel.VitGpt2);

        Assert.Null(hub.Captioner);
        Assert.Contains(hub.LoadErrors, e => e.StartsWith("ViT-GPT2:", StringComparison.Ordinal));
    }
}
