using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// CLIPSeg (CLIP-UNet) zero-shot image segmentation, run in-process via ONNX.
///
/// A single UNet takes the crop's concept text (token ids + attention mask, produced
/// by <see cref="ClipTokenizer"/> from vocab.json + merges.txt) and the 352x352 image
/// and returns a 352x352 mask. Because the prompt is textual, the detected box is
/// ignored &mdash; the concept label alone drives segmentation.
///
/// Pre-processing (matches the ONNX reference): stretch-resize to 352x352, RGB bytes
/// / 255, no mean/std, input shape [1,3,352,352]; token ids and attention mask are
/// int64 [1, seq] (the graph declares both as rank-2 [batch, sequence_length]).
/// Output: the segmentation mask (brighter = in-concept).
///
/// Model I/O names are matched defensively (by element type / rank / name) so minor
/// ONNX export variants still work, and the output that is the mask is preferred by
/// name (mask) with a positional fallback.
/// </summary>
public sealed class ClipSegSegmenter : IMaskGenerator
{
    private const int Side = 352;

    public string Name => "CLIPSeg";
    public int MaskSize => Side;

    private readonly InferenceSession? _session;
    private readonly string _modelPath; // for the lazy CPU fallback session
    private readonly string _imageIn;
    private readonly string? _tokenIn;
    private readonly string? _attentionIn;
    private readonly string? _maskOut;
    private readonly ClipTokenizer? _tokenizer;

    private readonly object _fallbackGate = new();
    private InferenceSession? _cpuFallback;

    private ClipSegSegmenter(
        InferenceSession session, string modelPath, string imageIn, string? tokenIn,
        string? attentionIn, string? maskOut, ClipTokenizer? tokenizer)
    {
        _session = session;
        _modelPath = modelPath;
        _imageIn = imageIn;
        _tokenIn = tokenIn;
        _attentionIn = attentionIn;
        _maskOut = maskOut;
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// Attempts to load the UNet (unet.onnx / model.onnx / the largest .onnx in the
    /// folder) plus the CLIP tokenizer (vocab.json + merges.txt). Returns null with a
    /// reason in <paramref name="error"/> when either is missing or malformed.
    /// </summary>
    public static ClipSegSegmenter? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;

        string? onnxFile = FirstExisting(modelDir, "unet.onnx", "model.onnx")
            ?? SafeList(modelDir, "*.onnx").OrderByDescending(SafeLength).Cast<string?>().FirstOrDefault();
        if (onnxFile is null)
        {
            error = "no CLIPSeg ONNX found in " + modelDir;
            return null;
        }

        string modelPath = Path.Combine(modelDir, onnxFile);
        InferenceSession session;
        try
        {
            session = OnnxFactory.Create(modelPath, useGpu);
        }
        catch (Exception ex)
        {
            error = "could not create the CLIPSeg session for " + onnxFile + ": " + ex.Message;
            return null;
        }

        string? imageIn = null, tokenIn = null, attentionIn = null;
        foreach (var im in session.InputMetadata)
        {
            string name = im.Key;
            int rank = im.Value.Dimensions.Length;
            bool isInt = im.Value.ElementType.ToString().Contains("INT", StringComparison.OrdinalIgnoreCase);
            if (rank == 4 && imageIn is null) imageIn = name;
            else if (isInt && attentionIn is null && name.Contains("attention", StringComparison.OrdinalIgnoreCase)) attentionIn = name;
            else if (isInt && tokenIn is null) tokenIn = name;
        }
        if (imageIn is null)
        {
            session.Dispose();
            error = "could not find the image input on " + onnxFile + "; expected a [1,3,352,352] tensor";
            return null;
        }

        string maskOut =
            session.OutputMetadata.FirstOrDefault(o => o.Key.Contains("mask", StringComparison.OrdinalIgnoreCase)).Key
            ?? session.OutputMetadata.First().Key;

        // The tokenizer is required: CLIPSeg consumes token ids, not raw text.
        string vocab = Path.Combine(modelDir, "vocab.json");
        string merges = Path.Combine(modelDir, "merges.txt");
        if (!File.Exists(vocab) || !File.Exists(merges))
        {
            session.Dispose();
            error = "CLIPSeg tokenizer files (vocab.json, merges.txt) not found in " + modelDir;
            return null;
        }

        ClipTokenizer tokenizer;
        try { tokenizer = new ClipTokenizer(vocab, merges); }
        catch (Exception ex)
        {
            session.Dispose();
            error = "could not load the CLIPSeg tokenizer (vocab.json/merges.txt): " + ex.Message;
            return null;
        }

        return new ClipSegSegmenter(session, modelPath, imageIn, tokenIn, attentionIn, maskOut, tokenizer);
    }

    /// <inheritdoc/>
    public MaskState Prepare(Image<Rgba32> baseImage)
    {
        int baseW = baseImage.Width, baseH = baseImage.Height;
        // Stretch to 352x352, RGB / 255, no mean/std normalization.
        using var resized = ImagePreprocess.Resize(baseImage, Side, Side);
        byte[] rgb = ImagePreprocess.ToRgbBytes(resized);
        float[] tensor = ImagePreprocess.ToChwTensor(rgb, Side, Side, rescale: 1f / 255f);
        return new MaskState
        {
            MaskSize = Side,
            ScaleX = Side / (float)baseW,
            ScaleY = Side / (float)baseH,
            PadX = 0,
            PadY = 0,
            Payload = tensor,
        };
    }

    /// <inheritdoc/>
    public Image<Rgba32> Mask(MaskState state, string concept, ConceptBox? box)
    {
        if (_session is null) throw new InvalidOperationException("CLIPSeg session not loaded.");
        if (_tokenizer is null) throw new InvalidOperationException("CLIPSeg tokenizer not loaded.");
        float[] img = state.Payload as float[] ?? throw new InvalidOperationException("Prepare() was not called for this image.");

        var feeds = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_imageIn, new DenseTensor<float>(img, new[] { 1, 3, Side, Side })),
        };
        if (_tokenIn is not null && _attentionIn is not null)
        {
            var (ids, attn) = _tokenizer.Encode(concept);
            // Widen int -> long element-by-element: ids/Attn are int[] and the graph
            // inputs are int64 [1, seq]. (C# LINQ Cast<long> on a value type throws
            // InvalidCastException, and the session rejects rank-1 tensors.)
            feeds.Add(NamedOnnxValue.CreateFromTensor(_tokenIn, new DenseTensor<long>(ToLongs(ids), new[] { 1, ids.Length })));
            feeds.Add(NamedOnnxValue.CreateFromTensor(_attentionIn, new DenseTensor<long>(ToLongs(attn), new[] { 1, attn.Length })));
        }

        var (plane, mw, mh) = Run(feeds.ToArray());

        // The UNet output may already be a probability or a logit; squash either way
        // to a 0-1 mask.
        var maskImg = new Image<Rgba32>(mw, mh);
        int p = 0;
        maskImg.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < mh; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < mw; x++)
                {
                    float v = plane[p++];
                    float prob = v >= 0f && v <= 1f ? v : 1f / (1f + MathF.Exp(-v));
                    byte b = (byte)Math.Clamp(prob * 255f, 0, 255);
                    row[x] = new Rgba32(b, b, b, 255);
                }
            }
        });
        return maskImg;
    }

    /// <summary>
    /// Runs the graph on the primary session; if the GPU execution provider
    /// refuses the graph (this exported UNet's dynamic Squeeze node fails on
    /// DirectML with E_INVALIDARG, verified against the model), retries once on a
    /// lazily-created CPU session and keeps it for subsequent calls.
    /// </summary>
    private (float[] Plane, int Mw, int Mh) Run(NamedOnnxValue[] feeds)
    {
        try
        {
            return Extract(_session!.Run(feeds));
        }
        catch (OnnxRuntimeException ex)
        {
            var cpu = GetCpuFallback();
            if (cpu is null) throw;
            Console.Error.WriteLine("CLIPSeg: GPU run failed (" + ex.Message + "); falling back to CPU.");
            return Extract(cpu.Run(feeds));
        }
    }

    private InferenceSession? GetCpuFallback()
    {
        lock (_fallbackGate)
        {
            if (_cpuFallback is not null) return _cpuFallback;
            try { _cpuFallback = OnnxFactory.Create(_modelPath, useGpu: false); }
            catch { _cpuFallback = null; }
            return _cpuFallback;
        }
    }

    private (float[] Plane, int Mw, int Mh) Extract(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        using (outputs)
        {
            var maskOutVal = outputs.FirstOrDefault(o => o.Name == _maskOut) ?? outputs.First();
            var mask = maskOutVal.AsTensor<float>();

            int[] dims = mask.Dimensions.ToArray();
            int mw = dims[^1], mh = dims[^2];
            // Take the first [H,W] plane regardless of leading batch/channel dims.
            float[] plane = new float[mh * mw];
            int idx = 0;
            foreach (var v in mask)
            {
                if (idx >= plane.Length) break;
                plane[idx++] = v;
            }
            return (plane, mw, mh);
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
        _cpuFallback?.Dispose();
    }

    private static long[] ToLongs(int[] ints)
    {
        var longs = new long[ints.Length];
        for (var i = 0; i < ints.Length; i++) longs[i] = ints[i];
        return longs;
    }

    private static string? FirstExisting(string dir, params string[] names)
    {
        foreach (var n in names)
            if (n is not null && File.Exists(Path.Combine(dir, n))) return n;
        return null;
    }

    private static IEnumerable<string> SafeList(string dir, string pattern)
    {
        try { return Directory.EnumerateFiles(dir, pattern).ToArray(); }
        catch (Exception) { return Array.Empty<string>(); }
    }

    private static long SafeLength(string file)
    {
        try { return new FileInfo(file).Length; }
        catch { return 0; }
    }
}
