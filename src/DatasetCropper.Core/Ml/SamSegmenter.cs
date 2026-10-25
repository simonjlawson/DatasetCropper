using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// A name-keyed encoder output tensor, ready to be fed to the decoder.
/// </summary>
public readonly record struct SamFeat(float[] Data, int[] Shape);

/// <summary>
/// The result of encoding a base image with SAM's image encoder, plus the
/// coordinate mapping between base-image pixels and SAM's 1024x1024 space.
///
/// Encoders are name-driven: SAM 1 exports a single <c>image_embeddings</c>
/// tensor, while SAM 2 (hierra_base_plus) exports <c>image_embed</c> plus
/// two extra high-res feature maps (<c>high_res_feats_0</c>,
/// <c>high_res_feats_1</c>). The decoder expects all of them.
///
/// <see cref="Feats"/> stores each encoder output by name and the decoder
/// feed-loop matches input names against this dictionary so the wiring is
/// resilient across export variants.
/// </summary>
public sealed class SamEncoding
{
    /// <summary>
    /// Encoder output tensors keyed by their onnx output name.
    /// </summary>
    public required IReadOnlyDictionary<string, SamFeat> Feats { get; init; }

    /// <summary>baseW, baseH in base-image pixels.</summary>
    public int BaseWidth { get; init; }
    public int BaseHeight { get; init; }

    /// <summary>scale = 1024 / max(baseW, baseH).</summary>
    public float Scale { get; init; }

    /// <summary>Pad offsets of the base image inside the 1024x1024 SAM canvas.</summary>
    public int PadX { get; init; }
    public int PadY { get; init; }

    public const int SamSize = 1024;

    /// <summary>
    /// The 1024x1024 canvas geometry (scale + center padding) a base image of
    /// the given size would be resized to inside it — scale = size / max(baseW,
    /// baseH), image centered. Shared by <see cref="SamSegmenter.Encode"/> and by
    /// callers that need the mapping without an encoding (SAM not loaded).
    /// </summary>
    public static (float Scale, int PadX, int PadY) CanvasGeometry(int baseW, int baseH)
    {
        float scale = SamSize / (float)Math.Max(baseW, baseH);
        int newW = Math.Max(1, (int)Math.Round(baseW * scale));
        int newH = Math.Max(1, (int)Math.Round(baseH * scale));
        return (scale, (SamSize - newW) / 2, (SamSize - newH) / 2);
    }

    /// <summary>Maps a base-image point to SAM-space.</summary>
    public (float X, float Y) ToSamSpace(float bx, float by) =>
        (bx * Scale + PadX, by * Scale + PadY);

    /// <summary>
    /// Maps a base-image rectangle to its SAM-space rectangle (int coords).
    /// </summary>
    public (int X, int Y, int W, int H) ToSamRect(int bx, int by, int bw, int bh)
    {
        int x = (int)Math.Round(bx * Scale + PadX);
        int y = (int)Math.Round(by * Scale + PadY);
        int w = Math.Max(1, (int)Math.Round(bw * Scale));
        int h = Math.Max(1, (int)Math.Round(bh * Scale));
        return (x, y, w, h);
    }
}

/// <summary>
/// Segment Anything Model, run in-process via ONNX as two models:
///   image_encoder.onnx  (1,3,1024,1024) -> one or more feature tensors
///   mask_decoder.onnx   point-prompt decoder -> low-res or full-res mask
///
/// Supports both the SAM 1 (ViT-B) layout (single <c>image_embeddings</c>
/// output + an <c>orig_im_size</c> decoder input) and the SAM 2 layout
/// (an <c>image_embed</c> plus two <c>high_res_feats_*</c> outputs, no
/// <c>orig_im_size</c> input). The feed loop inspects the decoder's declared
/// inputs and serves each from either the encoder output (matched by name)
/// or a synthesized prompt tensor, so the code adapts to whichever variant
/// is on disk.
/// </summary>
public sealed class SamSegmenter : IMaskGenerator
{
    private readonly InferenceSession? _encoder;
    private readonly InferenceSession? _decoder;
    private readonly string _encIn;
    private readonly string[] _decInputNames;
    private readonly string _decMaskOut;

    private SamSegmenter(InferenceSession? encoder, InferenceSession? decoder,
        string encIn, string[] decInputNames, string decMaskOut)
    {
        _encoder = encoder;
        _decoder = decoder;
        _encIn = encIn;
        _decInputNames = decInputNames;
        _decMaskOut = decMaskOut;
    }

    public static SamSegmenter? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string encPath = Path.Combine(modelDir, "image_encoder.onnx");
        string decPath = Path.Combine(modelDir, "mask_decoder.onnx");
        if (!File.Exists(encPath) || !File.Exists(decPath))
        {
            error = "image_encoder.onnx and/or mask_decoder.onnx not found in " + modelDir;
            return null;
        }
        var encoder = OnnxFactory.Create(encPath, useGpu);
        var decoder = OnnxFactory.Create(decPath, useGpu);

        string encIn = encoder.InputNames[0];
        var decInputNames = decoder.InputNames.ToArray();

        var decOutputs = decoder.OutputNames;
        string decMaskOut = decOutputs.FirstOrDefault(n => n.Contains("low_res", StringComparison.OrdinalIgnoreCase))
                           ?? decOutputs.FirstOrDefault(n => n.Contains("mask", StringComparison.OrdinalIgnoreCase))
                           ?? decOutputs[0];

        return new SamSegmenter(encoder, decoder, encIn, decInputNames, decMaskOut);
    }

    public bool IsLoaded => _encoder is not null && _decoder is not null;

    public string Name => "SAM";
    public int MaskSize => SamEncoding.SamSize;

    /// <summary>
    /// Encodes the base image once, capturing every encoder output (SAM 1
    /// emits <c>image_embeddings</c>; SAM 2 emits <c>image_embed</c>,
    /// <c>high_res_feats_0</c> and <c>high_res_feats_1</c>).
    /// </summary>
    public SamEncoding Encode(Image<Rgba32> baseImage)
    {
        if (_encoder is null) throw new InvalidOperationException("SAM encoder not loaded.");
        int baseW = baseImage.Width, baseH = baseImage.Height;
        (float scale, int padX, int padY) = SamEncoding.CanvasGeometry(baseW, baseH);
        int newW = Math.Max(1, (int)Math.Round(baseW * scale));
        int newH = Math.Max(1, (int)Math.Round(baseH * scale));

        var resized = ImagePreprocess.Resize(baseImage, newW, newH);
        byte[] rgb = ImagePreprocess.ToRgbBytes(resized, (0, 0, 0)); // black under the image
        // Center on a zero canvas (SAM pads with zeros).
        var canvas = ImagePreprocess.NewFilled(SamEncoding.SamSize, SamEncoding.SamSize, 0, 0, 0);
        // Draw the (newW,newH) child at (padX,padY); it is already exactly that size.
        canvas.Mutate(m => m.DrawImage(ImagePreprocess.NewFromBytes(rgb, newW, newH),
            new Point(padX, padY), 1f));
        byte[] canvasBytes = ImagePreprocess.ToRgbBytes(canvas, (0, 0, 0));

        float[] input = ImagePreprocess.ToChwTensor(
            canvasBytes, SamEncoding.SamSize, SamEncoding.SamSize, rescale: 1f / 255f,
            meanR: 0.485f, meanG: 0.456f, meanB: 0.406f,
            stdR: 0.229f, stdG: 0.224f, stdB: 0.225f);

        var inputTensor = new DenseTensor<float>(input, [1, 3, SamEncoding.SamSize, SamEncoding.SamSize]);
        using var outputs = _encoder.Run([NamedOnnxValue.CreateFromTensor(_encIn, inputTensor)]);

        var feats = new Dictionary<string, SamFeat>(StringComparer.OrdinalIgnoreCase);
        foreach (var ortVal in outputs)
        {
            var t = ortVal.AsTensor<float>();
            int len = (int)t.Length;
            var flat = new float[len];
            int n = 0;
            foreach (var v in t) flat[n++] = v;
            feats[ortVal.Name] = new SamFeat(flat, t.Dimensions.ToArray());
        }

        return new SamEncoding
        {
            Feats = feats,
            BaseWidth = baseW,
            BaseHeight = baseH,
            Scale = scale,
            PadX = padX,
            PadY = padY
        };
    }

    /// <summary>
    /// Implements <see cref="IMaskGenerator.Prepare"/>: encodes the base image once
    /// (the expensive image-encoder pass) and wraps the result in a
    /// <see cref="MaskState"/> whose payload is the <see cref="SamEncoding"/>.
    /// </summary>
    public MaskState Prepare(Image<Rgba32> baseImage)
    {
        SamEncoding enc = Encode(baseImage);
        return new MaskState
        {
            MaskSize = SamEncoding.SamSize,
            ScaleX = enc.Scale,
            ScaleY = enc.Scale,
            PadX = enc.PadX,
            PadY = enc.PadY,
            Payload = enc,
        };
    }

    /// <summary>
    /// Produces a 1024x1024 mask (in SAM-space) for a concept box, using the box
    /// center + two corners as positive point prompts.
    /// </summary>
    public Image<Rgba32> Mask(SamEncoding enc, ConceptBox box)
    {
        if (_decoder is null) throw new InvalidOperationException("SAM decoder not loaded.");
        if (enc.Feats.Count == 0) throw new InvalidOperationException("SamEncoding has no encoder features; call Encode() first.");

        // Box corners + center in SAM-space.
        var (sx1, sy1) = enc.ToSamSpace(box.X, box.Y);
        var (sx2, sy2) = enc.ToSamSpace(box.X2, box.Y2);
        var (scx, scy) = enc.ToSamSpace(box.Cx, box.Cy);

        const int numPoints = 3;
        var coords = new float[numPoints * 2]
        {
            scx, scy, sx1, sy1, sx2, sy2
        };
        var labels = new float[numPoints] { 1, 1, 1 };
        var maskInput = new float[1 * 1 * 256 * 256]; // zeros
        var hasMask = new float[] { 0f };
        var origSize = new float[] { enc.BaseHeight, enc.BaseWidth };

        // Dispatch every declared decoder input:
        //   * If it matches an encoder output by name -> take it from enc.Feats.
        //   * Otherwise synthesize the prompt tensor (coords/labels/mask flags).
        // This covers both SAM 1 (6 inputs) and SAM 2 (7 inputs, no orig_im_size).
        var feeds = new List<NamedOnnxValue>(_decInputNames.Length);
        foreach (var inName in _decInputNames)
        {
            if (enc.Feats.TryGetValue(inName, out SamFeat feat))
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(feat.Data, feat.Shape)));
                continue;
            }
            string lower = inName.ToLowerInvariant();
            if (lower.Contains("has_mask"))
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(hasMask, new[] { 1 })));
            }
            else if (lower.Contains("point_coord") || lower is "coords" or "coordinate" or "box")
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(coords, new[] { 1, numPoints, 2 })));
            }
            else if (lower.Contains("point_label") || lower is "labels" or "label")
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(labels, new[] { 1, numPoints })));
            }
            else if (lower.Contains("mask_input"))
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(maskInput, new[] { 1, 1, 256, 256 })));
            }
            else if (lower.Contains("orig_im_size") || lower.Contains("orig_size"))
            {
                feeds.Add(NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<float>(origSize, new[] { 2 })));
            }
            // Unknown inputs (e.g. optional KV-cached state in some future exports)
            // are skipped silently: ONNX Runtime treats missing inputs as "use the
            // graph default", which is the correct behavior for optional state.
        }

        using var outputs = _decoder.Run(feeds.ToArray());
        var maskOut = outputs.FirstOrDefault(o => o.Name == _decMaskOut)?.AsTensor<float>()
                       ?? throw new InvalidOperationException("mask output missing");

        int[] mdims = maskOut.Dimensions.ToArray();
        int mw = mdims[^1];
        int mh = mdims[^2];
        // Take the first H×W slice, regardless of how many leading batch/mask dims
        // the export uses (SAM 1 returns [1, 1, 256, 256]; SAM 2 typically
        // [3, 256, 256]).
        var lowRes = new float[mh * mw];
        int idx = 0;
        foreach (var v in maskOut.AsEnumerable())
        {
            if (idx >= mh * mw) break;
            lowRes[idx++] = v;
        }

        // Upsample to 1024x1024.
        var maskImg = new Image<Rgba32>(mw, mh);
        maskImg.ProcessPixelRows(accessor =>
        {
            int p = 0;
            for (int y = 0; y < mh; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < mw; x++)
                {
                    // The decoder output is a *logit* (pre-sigmoid score), exactly
                    // like the reference PyTorch implementation which applies
                    // F.sigmoid before upsampling. Without the sigmoid the raw
                    // logits clamp to 0/255 and the mask is a hard binary blob.
                    float logit = lowRes[p++];
                    float prob = 1f / (1f + MathF.Exp(-logit));
                    byte v = (byte)Math.Clamp(prob * 255f, 0, 255);
                    row[x] = new Rgba32(v, v, v, 255);
                }
            }
        });
        // Catmull-Rom instead of the default nearest-neighbor so the 256->1024
        // upscale interpolates smoothly rather than producing 4x4 blocks.
        return ImagePreprocess.Resize(maskImg, SamEncoding.SamSize, SamEncoding.SamSize,
            KnownResamplers.CatmullRom);
    }

    /// <summary>
    /// Implements <see cref="IMaskGenerator.Mask"/>: decodes the requested concept's
    /// box. The <paramref name="concept"/> string is accepted for interface symmetry
    /// but is not used by SAM (the box is the prompt). A null box is an error.
    /// </summary>
    public Image<Rgba32> Mask(MaskState state, string concept, ConceptBox? box)
    {
        SamEncoding enc = state.Payload as SamEncoding
            ?? throw new InvalidOperationException("Prepare() was not called for this image.");
        if (box is null) throw new InvalidOperationException("SAM requires a detected box (point prompt) to segment.");
        return Mask(enc, box);
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        _decoder?.Dispose();
    }
}
