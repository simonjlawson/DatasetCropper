using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;

namespace DatasetCropper.Ml;

/// <summary>
/// Downloads missing ONNX model files into the configured models folder.
///
/// URL sources (kept here so they can be redirected to a preferred mirror):
///   - WD14  : haomurui/wd-DatasetCropper-v2 (model.onnx, selected_tags.csv)
///   - ViT-GPT2: Xenova/vit-gpt2-image-captioning (quantized encoder/decoder + GPT-2 tokenizer)
///   - DINO  : onnx-community/grounding-dino-tiny-ONNX (onnx/model.onnx, tokenizer.json)
///   - SAM   : vietanhdev/segment-anything-2-onnx-models
///             (sam2_hiera_base_plus.encoder.onnx, sam2_hiera_base_plus.decoder.onnx)
///             NOTE: SAM 2 runtime wiring in SamSegmenter still expects SAM 1 I/O —
///             the downloader fetches SAM 2 weights but inference will need to be
///             re-wired before those files will run end-to-end.
/// Swap any constant below for a local mirror if preferred.
///
/// No integrity check is performed (files are written atomically via a .part
/// temp file so a cancelled/failed download never leaves a truncated model
/// in place).
/// </summary>
public static class ModelDownloader
{
    private static readonly HttpClient Http = new() { Timeout = System.Threading.Timeout.InfiniteTimeSpan };

    private static readonly (string Name, string Sub, string File, string Url)[] Sources =
    {
        ("WD14", "wd14", "model.onnx",
            "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-DatasetCropper-v2/resolve/main/model.onnx"),
        ("WD14", "wd14", "selected_tags.csv",
            "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-DatasetCropper-v2/raw/main/selected_tags.csv"),
        ("ViT-GPT2", "vit-gpt2", "encoder_model_quantized.onnx",
            "https://huggingface.co/Xenova/vit-gpt2-image-captioning/resolve/main/onnx/encoder_model_quantized.onnx"),
        ("ViT-GPT2", "vit-gpt2", "decoder_model_quantized.onnx",
            "https://huggingface.co/Xenova/vit-gpt2-image-captioning/resolve/main/onnx/decoder_model_quantized.onnx"),
        ("ViT-GPT2", "vit-gpt2", "vocab.json",
            "https://huggingface.co/Xenova/vit-gpt2-image-captioning/raw/main/vocab.json"),
        ("ViT-GPT2", "vit-gpt2", "merges.txt",
            "https://huggingface.co/Xenova/vit-gpt2-image-captioning/raw/main/merges.txt"),
        ("DINO", "grounding-dino", "model.onnx",
            "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model.onnx"),
        ("DINO", "grounding-dino", "tokenizer.json",
            "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/tokenizer.json"),
        ("SAM", "sam", "image_encoder.onnx",
            "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_base_plus.encoder.onnx"),
        ("SAM", "sam", "mask_decoder.onnx",
            "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_base_plus.decoder.onnx"),
        ("CLIPSeg", "clipseg", "model.onnx",
            "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/main/onnx/model.onnx"),
        ("CLIPSeg", "clipseg", "merges.txt",
            "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/main/merges.txt"),
        ("CLIPSeg", "clipseg", "vocab.json",
            "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/main/vocab.json"),
        ("Rembg-human", "rembg", "u2net_human_seg.onnx",
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"),

        ("YOLO", "yolo", "yolov8n.onnx",
            "https://huggingface.co/Calvin-ML/YOLOv8/resolve/main/yolov8n.onnx"),
    };

    /// <summary>
    /// Returns one entry per file that is missing on disk. Empty list means
    /// everything is present (per the file-name list in this table).
    /// </summary>
    public static List<(string Name, string Sub, string File, string Url)> FindMissing(string modelsFolder)
    {
        var missing = new List<(string, string, string, string)>();
        foreach (var (name, sub, file, url) in Sources)
        {
            if (!File.Exists(Path.Combine(modelsFolder, sub, file)))
                missing.Add((name, sub, file, url));
        }
        return missing;
    }

    /// <summary>
    /// Downloads every missing file into &lt;modelsFolder&gt;/&lt;sub&gt;/&lt;file&gt;,
    /// reporting progress via <paramref name="log"/>. Files already present
    /// are skipped. Errors are logged per file and the method always returns
    /// with whatever succeeded — partial results are visible on disk and can
    /// be verified via <see cref="FindMissing"/>.
    /// </summary>
    public static int Download(string modelsFolder, Action<string> log, CancellationToken token)
        => Download(modelsFolder, log, _ => { }, token);

    /// <summary>
    /// As <see cref="Download(string, Action{string}, CancellationToken)"/> but
    /// additionally reports a live text progress bar for the current file via
    /// <paramref name="progress"/>. That callback is invoked in place (replacing
    /// the same line) several times per second and is not written to the log
    /// file, so it carries no file I/O.
    ///
    /// Each file's <i>final</i> call is its outcome: an <c>OK …</c> or
    /// <c>FAILED …</c> line that replaces the last bar frame. That is the line's
    /// final at-rest state, so the download never leaves a stale 100% bar above a
    /// separate result line — one line per file, end to end.
    /// </summary>
    public static int Download(string modelsFolder, Action<string> log, Action<string> progress, CancellationToken token)
    {
        int downloaded = 0;
        foreach (var (name, sub, file, url) in Sources)
        {
            string dir = Path.Combine(modelsFolder, sub);
            string dest = Path.Combine(dir, file);
            if (File.Exists(dest)) continue;
            token.ThrowIfCancellationRequested();
            log($"  Downloading {name} {file} ...");
            progress(FormatProgress(name, file, 0, null, 0));
            try
            {
                Directory.CreateDirectory(dir);
                string part = dest + ".part";
                if (File.Exists(part)) File.Delete(part);
                using HttpResponseMessage resp = Http
                    .GetAsync(url, HttpCompletionOption.ResponseHeadersRead, token)
                    .GetAwaiter().GetResult();
                if (!resp.IsSuccessStatusCode)
                {
                    progress($"    FAILED: HTTP {(int)resp.StatusCode} {resp.StatusCode} for {url}");
                    continue;
                }
                long? length = resp.Content.Headers.ContentLength;
                using Stream net = resp.Content.ReadAsStreamAsync(token).GetAwaiter().GetResult();
                using (var fs = File.Create(part))
                {
                    var sw = Stopwatch.StartNew();
                    long lastReport = 0;
                    CopyToAsync(net, fs, 1 << 20, w =>
                    {
                        // Throttle to ~5 updates/sec so the UI isn't flooded.
                        if (sw.ElapsedMilliseconds - lastReport < 200) return;
                        lastReport = sw.ElapsedMilliseconds;
                        progress(FormatProgress(name, file, w, length, sw.Elapsed.TotalSeconds));
                    }, token).GetAwaiter().GetResult();
                }
                long written = new FileInfo(part).Length;
                File.Move(part, dest, overwrite: true);
                downloaded++;
                // One line per file, end to end: the outcome replaces the last
                // bar frame in place so the log never shows a stale 100% bar
                // followed by a separate OK line.
                progress($"    OK {name} {file} ({FormatBytes(written)}" + (length is null ? "" : $", total {FormatBytes(length.Value)}") + ")");
            }
            catch (OperationCanceledException)
            {
                log("    Cancelled.");
                token.ThrowIfCancellationRequested();
            }
            catch (Exception ex)
            {
                progress($"    FAILED: {ex.Message}");
            }
        }
        return downloaded;
    }

    /// <summary>
    /// Builds one line of text progress bar, e.g.
    /// <c>SAM encoder 37% ▓▓▓▒▒▒▒▒▒ 900 MB / 2.4 GB (54.2 MB/s)</c>.
    /// When <paramref name="total"/> or <paramref name="elapsedSeconds"/> is
    /// unknown the bar still renders with whatever is known (a "live" bar of
    /// just the done count, no percentage or ETA).
    /// </summary>
    private static string FormatProgress(string name, string file, long done, long? total, double elapsedSeconds)
    {
        const int width = 24;
        bool known = total is > 0;
        double pct = known ? Math.Min(100d, 100d * done / total!.Value) : 0d;
        int filledBlocks = known ? (int)Math.Round(width * pct / 100d) : (done > 0 ? 1 : 0);
        filledBlocks = Math.Clamp(filledBlocks, 0, width);

        string bar = new string('\u2593', filledBlocks) + new string('\u2591', width - filledBlocks);
        string label = name == "SAM" && file.EndsWith(".onnx", StringComparison.Ordinal)
            ? "SAM " + file[..^6]
            : name;

        string body = known
            ? $"{pct:0}% {bar} {FormatBytes(done)} / {FormatBytes(total!.Value)}"
            : $"{bar} {FormatBytes(done)}";

        if (elapsedSeconds > 0 && done > 0)
            body += $" ({FormatBytes((long)(done / elapsedSeconds))}/s)";

        return $"  {label}  {body}";
    }

    /// <summary>
    /// Copies <paramref name="src"/> into <paramref name="dst"/> in fixed-size
    /// chunks, invoking <paramref name="onProgress"/> with the running byte
    /// count as it goes. Cancellation propagates from the inner loop as well.
    /// </summary>
    private static async Task CopyToAsync(
        Stream src, Stream dst, int bufferSize,
        Action<long> onProgress, CancellationToken token)
    {
        byte[] buffer = new byte[bufferSize];
        long written = 0;
        int n;
        while ((n = await src.ReadAsync(buffer.AsMemory(0, buffer.Length), token).ConfigureAwait(false)) > 0)
        {
            await dst.WriteAsync(buffer.AsMemory(0, n), token).ConfigureAwait(false);
            written += n;
            onProgress(written);
        }
    }

    private static string FormatBytes(long b) => b >= 1 << 20 ? $"{b / 1_000_000d:0.0} MB" : $"{b / 1_000d:0} KB";
}
