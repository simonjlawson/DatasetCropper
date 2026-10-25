using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace DatasetCropper.Ml;

/// <summary>
/// Low-level pixel &amp; tensor helpers shared by the ONNX models.
/// Uses only the core ImageSharp 2.x API (Resize / Crop / DrawImage-at-point)
/// so it is stable across patch versions.
/// </summary>
public static class ImagePreprocess
{
    /// <summary>
    /// Composites an image's alpha channel over a solid color and returns the
    /// result as a packed byte array in RGB order (row-major, no padding).
    /// </summary>
    public static byte[] ToRgbBytes(Image<Rgba32> image, (int r, int g, int b)? background = null)
    {
        var (br, bg, bb) = background ?? (255, 255, 255);
        var buf = new byte[image.Width * image.Height * 3];
        int idx = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < image.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    var p = row[x];
                    float a = p.A / 255f;
                    buf[idx++] = (byte)Math.Clamp(p.R * a + br * (1 - a), 0, 255);
                    buf[idx++] = (byte)Math.Clamp(p.G * a + bg * (1 - a), 0, 255);
                    buf[idx++] = (byte)Math.Clamp(p.B * a + bb * (1 - a), 0, 255);
                }
            }
        });
        return buf;
    }

    /// <summary>
    /// Builds a CHW float32 tensor of shape (1,3,H,W) from RGB bytes, applying
    /// an optional rescale (e.g. /255) and ImageNet-style mean/std normalization.
    /// </summary>
    public static float[] ToChwTensor(
        byte[] rgbBytes, int width, int height,
        float rescale = 1f,
        float? meanR = null, float? meanG = null, float? meanB = null,
        float? stdR = null, float? stdG = null, float? stdB = null)
    {
        var tensor = new float[3 * width * height];
        int plane = width * height;
        for (int i = 0; i < plane; i++)
        {
            float r = rgbBytes[i * 3] * rescale;
            float g = rgbBytes[i * 3 + 1] * rescale;
            float b = rgbBytes[i * 3 + 2] * rescale;
            tensor[i] = Normalize(r, meanR, stdR);
            tensor[plane + i] = Normalize(g, meanG, stdG);
            tensor[2 * plane + i] = Normalize(b, meanB, stdB);
        }
        return tensor;

        static float Normalize(float v, float? mean, float? std) =>
            mean is null || std is null ? v : (v - mean.Value) / std.Value;
    }

    /// <summary>
    /// Wraps packed RGB bytes (row-major, 3 bytes/pixel) into an opaque
    /// <see cref="Image{Rgba32}"/>.
    /// </summary>
    public static Image<Rgba32> NewFromBytes(byte[] rgb, int w, int h)
    {
        var img = new Image<Rgba32>(w, h);
        int idx = 0;
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    row[x] = new Rgba32(rgb[idx], rgb[idx + 1], rgb[idx + 2], 255);
                    idx += 3;
                }
            }
        });
        return img;
    }

    /// <summary>Creates a new image of (w,h) filled with a solid color.</summary>
    public static Image<Rgba32> NewFilled(int w, int h, byte r, byte g, byte b)
    {
        return new Image<Rgba32>(w, h, new Rgba32(r, g, b, 255));
    }

    /// <summary>
    /// Resizes an image to (newW,newH). When <paramref name="resampler"/> is null
    /// the ImageSharp default (nearest-neighbor) is used, preserving the historical
    /// behavior of every existing call site.
    /// </summary>
    public static Image<Rgba32> Resize(Image<Rgba32> src, int newW, int newH, IResampler? resampler = null)
    {
        var img = src.Clone();
        if (resampler is IResampler r)
            img.Mutate(x => x.Resize(newW, newH, r));
        else
            img.Mutate(x => x.Resize(newW, newH));
        return img;
    }

    /// <summary>
    /// Crops a rectangle from an image and resizes to (newW,newH). Coordinates are
    /// clamped to the source bounds; the output is always exactly (newW,newH).
    /// <paramref name="resampler"/> defaults to null (nearest-neighbor) so image
    /// crops keep their existing behavior; pass a smooth resampler for masks.
    /// </summary>
    public static Image<Rgba32> CropAndResize(
        Image<Rgba32> src, int x, int y, int w, int h, int newW, int newH, IResampler? resampler = null)
    {
        int sx = Math.Clamp(x, 0, src.Width - 1);
        int sy = Math.Clamp(y, 0, src.Height - 1);
        int sw = Math.Clamp(w, 1, src.Width - sx);
        int sh = Math.Clamp(h, 1, src.Height - sy);

        var crop = src.Clone();
        crop.Mutate(x => x.Crop(new Rectangle(sx, sy, sw, sh)));
        if (resampler is IResampler r)
            crop.Mutate(x => x.Resize(newW, newH, r));
        else
            crop.Mutate(x => x.Resize(newW, newH));
        return crop;
    }

    /// <summary>
    /// Composites a rectangle (x,y,w,h) of <paramref name="src"/> onto a new
    /// (canvasW,canvasH) canvas that is otherwise filled with <paramref name="padColor"/>.
    /// Only the in-bounds intersection is copied, scaled uniformly (canvasW/w on both
    /// axes) and placed where it belongs in the canvas — so a region that runs past
    /// the image edges is <i>padded</i> instead of clamped. Clamping would slide the
    /// window to a skewed (non-proportional) rectangle, and stretching that back to
    /// canvas size distorts the aspect (e.g. the whole image squeezed into a square
    /// crop). <paramref name="resampler"/> (default null = nearest-neighbor) is
    /// applied to the slice, matching <see cref="CropAndResize"/>. When the whole
    /// region is inside the source, the result is the same crop <see cref="CropAndResize"/>
    /// produces. The region and canvas should be proportional (a square region on a
    /// square canvas) since a single scale is applied to both axes.
    /// </summary>
    public static Image<Rgba32> PaddedCrop(
        Image<Rgba32> src, int x, int y, int w, int h, int canvasW, int canvasH,
        (byte r, byte g, byte b) padColor, IResampler? resampler = null)
    {
        int ix = Math.Max(0, x);
        int iy = Math.Max(0, y);
        int iw = Math.Min(x + w, src.Width) - ix;
        int ih = Math.Min(y + h, src.Height) - iy;
        if (iw <= 0 || ih <= 0)
            return NewFilled(canvasW, canvasH, padColor.r, padColor.g, padColor.b);

        double scale = (double)canvasW / w;
        int dw = Math.Max(1, (int)Math.Round(iw * scale));
        int dh = Math.Max(1, (int)Math.Round(ih * scale));
        int dx = Math.Max(0, (int)Math.Round((ix - x) * scale));
        int dy = Math.Max(0, (int)Math.Round((iy - y) * scale));
        // Keep the composite inside the canvas (guards rounding overshoot).
        dw = Math.Min(dw, canvasW - dx);
        dh = Math.Min(dh, canvasH - dy);

        using (var slice = CropAndResize(src, ix, iy, iw, ih, dw, dh, resampler))
        {
            var canvas = NewFilled(canvasW, canvasH, padColor.r, padColor.g, padColor.b);
            canvas.Mutate(m => m.DrawImage(slice, new Point(dx, dy), 1f));
            return canvas;
        }
    }

    /// <summary>
    /// Draws <paramref name="child"/> onto a new (w,h) solid-color canvas at the
    /// given top-left position (child is assumed to already be the right size).
    /// </summary>
    public static Image<Rgba32> CompositeOnCanvas(Image<Rgba32> child, int canvasW, int canvasH,
        int x, int y, (byte r, byte g, byte b) color)
    {
        var canvas = NewFilled(canvasW, canvasH, color.r, color.g, color.b);
        canvas.Mutate(m => m.DrawImage(child, new Point(x, y), 1f));
        return canvas;
    }
}
