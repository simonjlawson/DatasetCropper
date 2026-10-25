using System.IO;
using DatasetCropper.Ml;
using DatasetCropper.Services;
using Xunit;

namespace DatasetCropper.Tests;

/// <summary>
/// Covers the GPU-availability reporting added for "log GPU success/failure".
/// These tests never create an ONNX session (so no model file is needed); they
/// only probe provider availability, which is safe on any machine.
/// </summary>
public class OnnxFactoryTests
{
    [Fact]
    public void ProbeGpu_ReturnsGpuPrefixedLine()
    {
        string line = OnnxFactory.ProbeGpu();
        Assert.StartsWith("GPU:", line);
        // Either a success line naming a provider, or a CPU-fallback line.
        Assert.True(line.Contains("execution provider") || line.Contains("running on CPU"),
            "Unexpected probe line: " + line);
    }

    [Fact]
    public void MlHub_Load_UseGpu_LogsGpuLineExactlyOnce()
    {
        string modelsDir = Path.Combine(Path.GetTempPath(), "DatasetCropper_gpu_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(modelsDir);
        try
        {
            var lines = new List<string>();
            var hub = MlHub.Load(modelsDir, useGpu: true, log: lines.Add);
            hub.Dispose();

            var gpuLines = lines.Where(l => l.StartsWith("GPU:")).ToList();
            Assert.Single(gpuLines);
        }
        finally
        {
            try { Directory.Delete(modelsDir, true); } catch { /* best effort */ }
        }
    }

    [Fact]
    public void MlHub_Load_NoGpu_DoesNotLogGpuLine()
    {
        string modelsDir = Path.Combine(Path.GetTempPath(), "DatasetCropper_nogpu_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(modelsDir);
        try
        {
            var lines = new List<string>();
            var hub = MlHub.Load(modelsDir, useGpu: false, log: lines.Add);
            hub.Dispose();

            Assert.DoesNotContain(lines, l => l.StartsWith("GPU:"));
        }
        finally
        {
            try { Directory.Delete(modelsDir, true); } catch { /* best effort */ }
        }
    }
}
