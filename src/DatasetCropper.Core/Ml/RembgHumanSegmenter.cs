using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// rembg human-seg (u2net-human-seg) &mdash; a fixed human/foreground remover, run
/// in-process via ONNX.
///
/// No text or box prompt: the model produces one foreground mask per image and the
/// same mask is used for every concept on that image (so for a person + object
/// subject the mask covers whichever foreground part the crop contains).
///
/// Pre-processing (rembg reference): stretch-resize to 320x320, RGB / 255, ImageNet
/// mean/std, input [1,3,320,320]. Post-processing: sigmoid the output to a 0-1
/// probability. The mask is returned in 320x320 canvas space.
/// </summary>
public sealed class RembgHumanSegmenter : IMaskGenerator
{
    private const int Side = 320;

    public string Name => "Rembg-Human";
    public int MaskSize => Side;

    private readonly InferenceSession? _session;
    private readonly string _imageIn;
    private readonly string? _maskOut;

    private RembgHumanSegmenter(InferenceSession session, string imageIn, string? maskOut)
    {
        _session = session;
        _imageIn = imageIn;
        _maskOut = maskOut;
    }

    /// <summary>
    /// Attempts to load the u2net-human-seg ONNX (u2net_human_seg.onnx / model.onnx /
    /// the largest .onnx in the folder). Returns null with a reason in
    /// <paramref name="error"/> otherwise.
    /// </summary>
    public static RembgHumanSegmenter? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string? onnxFile = FirstExisting(modelDir, "u2net_human_seg.onnx", "model.onnx")
            ?? SafeList(modelDir, "*.onnx").OrderByDescending(SafeLength).Cast<string?>().FirstOrDefault();
        if (onnxFile is null)
        {
            error = "no Rembg-human ONNX found in " + modelDir;
            return null;
        }

        InferenceSession session;
        try
        {
            session = OnnxFactory.Create(Path.Combine(modelDir, onnxFile), useGpu);
        }
        catch (Exception ex)
        {
            error = "could not create the Rembg-human session for " + onnxFile + ": " + ex.Message;
            return null;
        }

        string imageIn = session.InputMetadata
            .FirstOrDefault(im => im.Value.Dimensions.Length == 4).Key
            ?? session.InputMetadata.First().Key;

        // The foreground probability is the output with the largest spatial extent
        // (u2net variants emit [1,1,320,320] or [1,320,320]); otherwise fall back to
        // the first output.
        string maskOut = session.OutputMetadata
            .OrderByDescending(o => o.Value.Dimensions.Skip(1).Select(d => d < 0 ? 1 : d).Aggregate(1, (a, b) => a * b))
            .First().Key;

        return new RembgHumanSegmenter(session, imageIn, maskOut);
    }

    /// <inheritdoc/>
    public MaskState Prepare(Image<Rgba32> baseImage)
    {
        int baseW = baseImage.Width, baseH = baseImage.Height;
        // Stretch to 320x320, RGB / 255, ImageNet mean/std.
        using var resized = ImagePreprocess.Resize(baseImage, Side, Side);
        byte[] rgb = ImagePreprocess.ToRgbBytes(resized);
        float[] tensor = ImagePreprocess.ToChwTensor(rgb, Side, Side, rescale: 1f / 255f,
            meanR: 0.485f, meanG: 0.456f, meanB: 0.406f,
            stdR: 0.229f, stdG: 0.224f, stdB: 0.225f);
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

    /// <summary>
    /// Runs the model for <paramref name="concept"/>. The concept string is accepted
    /// for interface symmetry but ignored: rembg-human emits a single fixed
    /// foreground mask.
    /// </summary>
    public Image<Rgba32> Mask(MaskState state, string concept, ConceptBox? box)
    {
        if (_session is null) throw new InvalidOperationException("Rembg-human session not loaded.");
        float[] img = state.Payload as float[] ?? throw new InvalidOperationException("Prepare() was not called for this image.");

        using var outputs = _session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(_imageIn, new DenseTensor<float>(img, new[] { 1, 3, Side, Side }))
        });
        var outVal = outputs.FirstOrDefault(o => o.Name == _maskOut) ?? outputs.First();
        var mask = outVal.AsTensor<float>();

        int[] dims = mask.Dimensions.ToArray();
        int mw = dims[^1], mh = dims[^2];
        float[] plane = new float[mh * mw];
        int idx = 0;
        foreach (var v in mask)
        {
            if (idx >= plane.Length) break;
            plane[idx++] = v;
        }

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
                    float prob = 1f / (1f + MathF.Exp(-v)); // sigmoid to 0-1
                    byte b = (byte)Math.Clamp(prob * 255f, 0, 255);
                    row[x] = new Rgba32(b, b, b, 255);
                }
            }
        });
        return maskImg;
    }

    public void Dispose() => _session?.Dispose();

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
