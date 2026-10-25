using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DatasetCropper.Imaging;

/// <summary>
/// Near-duplicate crop detection via a perceptual difference hash (dHash): the
/// image is reduced to a 9x8 luminance grid and one bit is set for each adjacent
/// cell pair where the left cell is brighter. Identical (or nearly identical)
/// crops give identical (or nearly identical) hashes; genuinely different
/// content gives a large Hamming distance; re-sampling the same content at
/// another pixel size changes only a few bits.
/// </summary>
public static class DuplicateDetector
{
    /// <summary>64-bit difference hash of the image (independent of its pixel size).</summary>
    public static ulong DHash(Image<Rgba32> image)
    {
        var grid = DownsampleLuma(image, 9, 8);
        ulong hash = 0;
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                if (grid[y, x] > grid[y, x + 1])
                    hash |= 1UL << (y * 8 + x);
            }
        }
        return hash;
    }

    /// <summary>Hamming distance of two hashes - how many of the 64 bits differ.</summary>
    public static int Distance(ulong a, ulong b) => System.Numerics.BitOperations.PopCount(a ^ b);

    /// <summary>
    /// Luma grid of (w x h) cell averages sampled from the source image.
    /// Block averaging keeps the result stable across pixel sizes without
    /// depending on a particular resampler.
    /// </summary>
    private static float[,] DownsampleLuma(Image<Rgba32> image, int w, int h)
    {
        var grid = new float[h, w];
        image.ProcessPixelRows(accessor =>
        {
        for (int ty = 0; ty < h; ty++)
        {
            int y0 = ty * image.Height / h;
            int y1 = Math.Max(y0 + 1, (ty + 1) * image.Height / h);
            for (int tx = 0; tx < w; tx++)
            {
                int x0 = tx * image.Width / w;
                int x1 = Math.Max(x0 + 1, (tx + 1) * image.Width / w);
                long sum = 0;
                int count = 0;
                for (int y = y0; y < y1; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = x0; x < x1; x++)
                    {
                        Rgba32 p = row[x];
                        sum += (p.R * 299 + p.G * 587 + p.B * 114) / 1000;
                    }
                    count += x1 - x0;
                }
                grid[ty, tx] = count > 0 ? (float)(sum / (double)count) : 0f;
            }
        }
        });
        return grid;
    }
}

/// <summary>
/// Remembers the hashes of crops already emitted for one source image and flags
/// a new crop when it is within <see cref="MaxDistance"/> Hamming bits of an
/// earlier one ("near-duplicate").
/// </summary>
public sealed class DedupeTracker
{
    private readonly int _maxDistance;
    private readonly List<(ulong Hash, string Label)> _seen = new();

    /// <param name="maxDistance">Maximum Hamming distance (0..64) that still counts as a duplicate.</param>
    public DedupeTracker(int maxDistance) => _maxDistance = Math.Clamp(maxDistance, 0, 64);

    /// <summary>
    /// True when the crop is a near-duplicate of one already emitted for this
    /// source image. <paramref name="duplicateOf"/> carries the earlier crop's
    /// name (null when it is not a duplicate).
    /// </summary>
    public bool IsDuplicate(ulong hash, string label, out string? duplicateOf)
    {
        foreach (var (h, l) in _seen)
        {
            if (DuplicateDetector.Distance(hash, h) <= _maxDistance)
            {
                duplicateOf = l;
                return true;
            }
        }
        duplicateOf = null;
        return false;
    }

    /// <summary>Records an emitted crop as the reference for later comparisons.</summary>
    public void Remember(ulong hash, string label) => _seen.Add((hash, label));
}
