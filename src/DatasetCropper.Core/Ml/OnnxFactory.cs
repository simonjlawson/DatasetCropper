using Microsoft.ML.OnnxRuntime;
using System.Runtime.InteropServices;

namespace DatasetCropper.Ml;

/// <summary>
/// Creates ONNX Runtime sessions.
///
/// When <paramref name="useGpu"/> is true we try, in order:
///   1. CUDA (NVIDIA) execution provider
///   2. DirectML (any DirectX-capable GPU) execution provider
/// and fall back to CPU-only if neither can be appended. Provider failure is
/// non-fatal: a session that only runs on CPU is always a valid result, so we
/// never throw out of here for a missing GPU driver — we just log and continue.
/// </summary>
public static class OnnxFactory
{
    private static readonly bool IsWindows =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    public static InferenceSession Create(string modelPath, bool useGpu)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2)
        };

        if (useGpu)
        {
            if (TryAttachGpuProvider(options, out var detail) is null)
                System.Diagnostics.Debug.WriteLine(
                    "useGpu requested but no GPU execution provider could be attached; running on CPU."
                    + (detail is null ? "" : " (" + detail + ")"));
        }

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Attempts to attach a GPU execution provider to <paramref name="options"/>,
    /// trying CUDA first then DirectML. Returns the name of the provider that was
    /// attached ("CUDA" or "DirectML"), or null when no GPU provider could be
    /// attached (the session will run on CPU). On failure, <paramref name="detail"/>
    /// carries the reason(s) for the log.
    /// </summary>
    public static string? TryAttachGpuProvider(SessionOptions options, out string? detail)
    {
        detail = null;
        // 1) CUDA — only meaningful on Windows with an NVIDIA GPU + CUDA runtime.
        if (IsWindows)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(0);
                return "CUDA";
            }
            catch (Exception ex)
            {
                detail = "CUDA: " + ex.Message;
            }
        }

        // 2) DirectML — works on any Windows GPU (NVIDIA/AMD/Intel).
        if (IsWindows)
        {
            try
            {
                options.AppendExecutionProvider_DML(0);
                return "DirectML";
            }
            catch (Exception ex)
            {
                detail = (detail is null ? "" : detail + "; ") + "DirectML: " + ex.Message;
            }
        }

        return null;
    }

    /// <summary>
    /// Probes which GPU execution provider (if any) is available on this machine,
    /// without creating a session. Returns a user-facing log line describing the
    /// outcome, e.g. "GPU: using CUDA execution provider." or
    /// "GPU: requested but no GPU execution provider available — running on CPU. (…)".
    /// </summary>
    public static string ProbeGpu()
    {
        var probe = new SessionOptions();
        string? provider = TryAttachGpuProvider(probe, out string? detail);
        return provider is null
            ? "GPU: requested but no GPU execution provider available — running on CPU."
                + (detail is null ? "" : " (" + detail + ")")
            : "GPU: using " + provider + " execution provider.";
    }
}
