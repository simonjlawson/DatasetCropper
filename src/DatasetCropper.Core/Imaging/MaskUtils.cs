using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;
using DatasetCropper.Ml;

namespace DatasetCropper.Imaging;

/// <summary>
/// Helpers for turning a full-base-image mask (e.g. SAM output in 1024x1024
/// SAM-space) into the per-crop 1024x1024 grayscale mask that ships with each
/// training crop.
/// </summary>
public static class MaskUtils
{
    /// <summary>
    /// Extracts the crop region from a full-size mask image and resizes it to
    /// cropSize x cropSize. The mask is expected to already be aligned to the
    /// base image (same pixel grid as the base).
    /// </summary>
    /// <param name="resampler">
    /// Optional resampler for the crop→cropSize upscale. Pass
    /// <c>KnownResamplers.CatmullRom</c> for masks so the low-resolution SAM output
    /// is interpolated smoothly instead of nearest-neighbor (which produces the
    /// blocky, pixelated edges users see). Defaults to null = nearest-neighbor,
    /// matching the historical behavior.
    /// </param>
    public static Image<Rgba32> CropMask(Image<Rgba32> fullMask, int x, int y, int w, int h, int cropSize,
        IResampler? resampler = null)
    {
        return ImagePreprocess.CropAndResize(fullMask, x, y, w, h, cropSize, cropSize, resampler);
    }

    /// <summary>
    /// Applies a light Gaussian blur to a mask to smooth hard, pixelated edges.
    /// A small standard deviation (1.5–2.0) is enough to break up the blocky
    /// upsampling artifacts without materially eroding the mask boundary. The
    /// input mask is cloned, so the caller's image is not modified.
    /// </summary>
    public static Image<Rgba32> SmoothMask(Image<Rgba32> mask, float stdDeviation = 2.0f)
    {
        var result = mask.Clone();
        result.Mutate(m => m.GaussianBlur(stdDeviation));
        return result;
    }

    /// <summary>
    /// Thresholds a smoothed mask into a strict black/white mask. Smoothing is
    /// intentionally performed before this step so small jagged boundary pixels
    /// do not survive as isolated specks.
    /// </summary>
    public static Image<Rgba32> ToBinaryMask(Image<Rgba32> mask, byte threshold = 128)
    {
        var sourcePixels = new byte[mask.Width * mask.Height];
        int index = 0;
        mask.ProcessPixelRows(src =>
        {
            for (int y = 0; y < mask.Height; y++)
            {
                var row = src.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    sourcePixels[index++] = row[x].R;
            }
        });

        var result = new Image<Rgba32>(mask.Width, mask.Height);
        result.ProcessPixelRows(dst =>
        {
            index = 0;
            for (int y = 0; y < mask.Height; y++)
            {
                var targetRow = dst.GetRowSpan(y);
                for (int x = 0; x < targetRow.Length; x++)
                {
                    byte value = sourcePixels[index++] >= threshold ? (byte)255 : (byte)0;
                    targetRow[x] = new Rgba32(value, value, value, 255);
                }
            }
        });
        return result;
    }

    /// <summary>
    /// Converts a mask (any luminance representation) to a grayscale PNG-ready
    /// image where the concept region is white (255) and the background is black
    /// (0). Expects a single-channel or grayscale-ish source; here we treat the
    /// red channel as the mask signal.
    /// </summary>
    public static Image<Rgba32> ToGrayscaleMask(Image<Rgba32> mask)
    {
        var result = new Image<Rgba32>(mask.Width, mask.Height);
        var srcPixels = new byte[mask.Width * mask.Height];
        int idx = 0;
        mask.ProcessPixelRows(src =>
        {
            for (int y = 0; y < mask.Height; y++)
            {
                var row = src.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    srcPixels[idx++] = row[x].R;
            }
        });
        idx = 0;
        result.ProcessPixelRows(dst =>
        {
            for (int y = 0; y < result.Height; y++)
            {
                var row = dst.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    byte v = srcPixels[idx++];
                    row[x] = new Rgba32(v, v, v, 255);
                }
            }
        });
        return result;
    }

    /// <summary>
    /// Builds a solid (all-white) mask of the given size, used when no concept
    /// is assigned to a crop.
    /// </summary>
    public static Image<Rgba32> SolidMask(int size, byte value = 255)
    {
        return new Image<Rgba32>(size, size, new Rgba32(value, value, value, 255));
    }
}
