using System.Diagnostics;
using System.IO;
using System.Text;

namespace DatasetCropper.Services;

/// <summary>
/// Wraps the Topaz Photo AI CLI (tpai.exe) for upscaling / enhancement.
///
/// Invocation:  tpai.exe -o &lt;outDir&gt; --overwrite &lt;args...&gt; &lt;input1&gt; [&lt;input2&gt;...]
/// where &lt;args...&gt; are the user-configurable enhancement options (default "--upscale").
/// The CLI accepts any number of input files in one invocation, so a whole
/// queue can be upscaled in a single process instead of one process per image.
/// Exit codes:  0 success, 1 partial, 255 no valid files, 254 bad login, 253 bad args.
/// </summary>
public static class TopazService
{
    public const string DefaultTpaiPath =
        @"C:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe";

    /// <summary>Per-image time allowance; the total cap scales with the batch size.</summary>
    private const int PerImageCapMs = 1000 * 60 * 30; // 30 minutes
    private const int HardCapMs = 1000 * 60 * 24;     // never kill a large batch before 24 h

    /// <summary>
    /// Resolves the tpai.exe path from a user setting (empty = default install).
    /// Returns null when the executable cannot be found.
    /// </summary>
    public static string? Resolve(string configuredPath)
    {
        string path = string.IsNullOrWhiteSpace(configuredPath) ? DefaultTpaiPath : configuredPath;
        return File.Exists(path) ? path : null;
    }

    /// <summary>
    /// Runs the Topaz enhancement(s) <paramref name="args"/> on a single image into
    /// <paramref name="outDir"/>. Returns the path of the produced file, or null on
    /// failure. <paramref name="args"/> is the user-configurable enhancement portion of
    /// the command line (e.g. "--upscale", "--noise --sharpen"); -o/--overwrite and the
    /// input file are added here.
    /// </summary>
    public static string? Upscale4x(string tpaiPath, string inputPath, string outDir, string args,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var produced = Upscale4xBatch(tpaiPath, new[] { inputPath }, outDir, args, progress, ct);
        return produced is not null
            && produced.TryGetValue(Path.GetFileNameWithoutExtension(inputPath).ToLowerInvariant(), out string? p)
            ? p
            : null;
    }

    /// <summary>
    /// Upscales a batch of images in a single tpai process (the CLI accepts any number
    /// of input files). Returns a map of <c>lowercased filename-without-extension ->
    /// produced file path</c> containing every input that got an output file, or null
    /// when the process could not be run at all (timeout, cancelled, crash) — the
    /// caller then falls back to the originals.
    /// </summary>
    public static Dictionary<string, string>? Upscale4xBatch(string tpaiPath,
        IReadOnlyList<string> inputPaths, string outDir, string args,
        IProgress<string>? progress = null, CancellationToken ct = default)
    {
        if (inputPaths.Count == 0)
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        int? code = RunTpai(tpaiPath, inputPaths, outDir, args, progress, ct);
        if (code is null) return null; // timeout, cancelled, or crash - caller falls back

        var produced = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string input in inputPaths)
        {
            string baseName = Path.GetFileNameWithoutExtension(input);
            string? candidate = Directory.EnumerateFiles(outDir)
                .Where(f => Path.GetFileNameWithoutExtension(f).Equals(baseName, StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(File.GetLastWriteTime)
                .FirstOrDefault();
            if (candidate is not null)
                produced[baseName.ToLowerInvariant()] = candidate;
        }
        return produced;
    }

    /// <summary>
    /// Spawns tpai for the whole batch and returns the exit code, or null when the
    /// process timed out (killed), was cancelled, or failed to start.
    /// </summary>
    private static int? RunTpai(string tpaiPath, IReadOnlyList<string> inputPaths,
        string outDir, string args, IProgress<string>? progress, CancellationToken ct)
    {
        try
        {
            // Topaz's CLI rejects relative paths for -o (exits -3 "Invalid output
            // path"), so resolve to an absolute path first.
            outDir = Path.GetFullPath(outDir);
            Directory.CreateDirectory(outDir);

            var psi = new ProcessStartInfo
            {
                FileName = tpaiPath,
                WorkingDirectory = Path.GetDirectoryName(tpaiPath) ?? "",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            psi.ArgumentList.Add("-o");
            psi.ArgumentList.Add(outDir);
            psi.ArgumentList.Add("--overwrite");
            // User-configurable enhancement args (whitespace-separated), then all
            // input files - one process for the whole batch.
            foreach (var a in args.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                psi.ArgumentList.Add(a);
            foreach (string input in inputPaths)
                psi.ArgumentList.Add(input);

            progress?.Report($"Topaz: upscaling {inputPaths.Count} image(s)");

            using var proc = new Process { StartInfo = psi };
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();
            proc.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) stderr.AppendLine(e.Data); };

            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();

            // Kill the process promptly on cancellation so we don't sit out the
            // full cap before the pipeline notices.
            CancellationTokenRegistration reg = ct.Register(() =>
            {
                try { proc.Kill(true); } catch { /* best effort */ }
            });

            int timeoutMs = Math.Min(HardCapMs, PerImageCapMs * inputPaths.Count);
            bool exited = proc.WaitForExit(timeoutMs);
            reg.Dispose();

            if (!exited)
            {
                try { proc.Kill(true); } catch { /* best effort */ }
                progress?.Report("Topaz: timed out");
                return null;
            }

            if (ct.IsCancellationRequested)
            {
                progress?.Report("Topaz: cancelled");
                return null;
            }

            int code = proc.ExitCode;
            progress?.Report(code == 0
                ? $"Topaz done: {inputPaths.Count} image(s)"
                : $"Topaz exited {code}: {stderr.ToString().Trim()}");
            return code;
        }
        catch (Exception ex)
        {
            progress?.Report("Topaz error: " + ex.Message);
            return null;
        }
    }
}
