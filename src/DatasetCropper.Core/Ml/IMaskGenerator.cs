using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// A mask-generator model (SAM, CLIPSeg, rembg-human, …). <see cref="Prepare"/> is
/// called once per base image and returns an opaque per-image state; <see cref="Mask"/>
/// then produces one full-frame mask (in the generator's own canvas space,
/// <c>MaskSize × MaskSize</c>) for a concept/region. Both calls are made under
/// the caller's model lock.
/// </summary>
public interface IMaskGenerator : IDisposable
{
    /// <summary>Display name ("SAM", "CLIPSeg", "Rembg-Human").</summary>
    string Name { get; }

    /// <summary>
    /// One-shot per-base-image preparation (e.g. running the image encoder, or a
    /// rembg-style model that only needs the pixels at inference time).
    /// </summary>
    MaskState Prepare(Image<Rgba32> baseImage);

    /// <summary>
    /// Produce the mask for <paramref name="concept"/>. <paramref name="box"/> is the
    /// detected box in base-image coordinates; generators that only take text (CLIPSeg)
    /// or ignore framing entirely (rembg) may disregard it. The result is a grayscale
    /// image sized <see cref="MaskSize"/> × <see cref="MaskSize"/>, brighter = in-concept.
    /// </summary>
    Image<Rgba32> Mask(MaskState state, string concept, ConceptBox? box);

    /// <summary>Side (in pixels) of the mask canvas this generator produces.</summary>
    int MaskSize { get; }
}

/// <summary>
/// Geometry mapping from base-image pixels into a generator's mask canvas plus the
/// generator-specific payload (encoder features, prepared pixels, …). The scale/pad
/// mapping is what lets a crop region be projected back out of canvas space.
/// </summary>
public sealed class MaskState
{
    public int MaskSize { get; init; }
    public float ScaleX { get; init; }
    public float ScaleY { get; init; }
    public int PadX { get; init; }
    public int PadY { get; init; }
    public object? Payload { get; init; }

    /// <summary>Project a base-image rect into canvas space (unclamped).</summary>
    public (int X, int Y, int W, int H) MaskRect(int x, int y, int w, int h)
        => ((int)MathF.Round(x * ScaleX) + PadX,
            (int)MathF.Round(y * ScaleY) + PadY,
            (int)MathF.Round(w * ScaleX),
            (int)MathF.Round(h * ScaleY));
}
