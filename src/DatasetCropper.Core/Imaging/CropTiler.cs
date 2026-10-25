using DatasetCropper.Models;

namespace DatasetCropper.Imaging;

/// <summary>
/// Produces overlapping 1024x1024 tile regions covering a base image, plus
/// concept-centered crops.
/// </summary>
public static class CropTiler
{
    public const int CropSize = 1024;
    public const int DefaultMinOverlap = 512; // 50% overlap

    /// <summary>
    /// Converts a minimum overlap (pixels) into the tile step: step =
    /// CropSize - overlap, clamped to [1, CropSize]. Larger overlap → smaller
    /// step → more tiles. overlap ≥ CropSize clamps to step 1 (dense tiling);
    /// overlap ≤ 0 clamps to step = CropSize (no overlap).
    /// </summary>
    public static int StepForOverlap(int minOverlap) =>
        Math.Clamp(CropSize - minOverlap, 1, CropSize);

    /// <summary>
    /// Computes the base image size: the input is resized so its smallest
    /// dimension becomes 1024, preserving aspect ratio.
    /// </summary>
    public static (int Width, int Height) BaseSize(int srcW, int srcH)
    {
        float scale = CropSize / (float)Math.Min(srcW, srcH);
        return (Math.Max(1, (int)Math.Round(srcW * scale)), Math.Max(1, (int)Math.Round(srcH * scale)));
    }

    /// <summary>
    /// Generates overlapping 1024x1024 tile regions for a base image of the given
    /// size. The last tile in each axis is clamped so it ends exactly at the
    /// right/bottom edge (no out-of-bounds, no gaps).
    /// </summary>
    public static List<CropRegion> Tile(int baseW, int baseH, int minOverlap = DefaultMinOverlap)
    {
        var regions = new List<CropRegion>();
        int step = StepForOverlap(minOverlap);
        var xs = AxisPositions(baseW, step);
        var ys = AxisPositions(baseH, step);
        foreach (int y in ys)
        {
            foreach (int x in xs)
            {
                regions.Add(new CropRegion
                {
                    X = x,
                    Y = y,
                    Width = Math.Min(CropSize, baseW - x),
                    Height = Math.Min(CropSize, baseH - y)
                });
            }
        }
        return regions;
    }

    private static List<int> AxisPositions(int length, int step)
    {
        var positions = new List<int>();
        if (length <= CropSize)
        {
            // Image is smaller than a tile in this axis: a single tile anchored at 0.
            positions.Add(0);
            return positions;
        }
        int lastStart = length - CropSize;
        for (int pos = 0; pos < lastStart; pos += step)
        {
            positions.Add(pos);
        }
        positions.Add(lastStart);
        return positions;
    }

    /// <summary>
    /// Maps a base-image-space subject box to the full-resolution square that
    /// will be cut from the original image: a 1:1 square of at least
    /// <paramref name="minimumSize"/> on a side (zooming in around the box when the box
    /// is small, keeping the box's own size when it is already larger), centered
    /// on the box when possible. The square is clamped so it always lies fully
    /// inside the image (the side length is capped at min(srcW, srcH) first),
    /// so no padding / black space ever appears in the crop.
    /// </summary>
    public static (int X, int Y, int Size) SubjectSquare(
        ConceptBox box, int baseW, int baseH, int srcW, int srcH, int minimumSize = CropSize)
    {
        float sx = (float)((double)srcW / baseW);
        float sy = (float)((double)srcH / baseH);
        int size = Math.Max(Math.Max(1, minimumSize), (int)Math.Round(box.W * sx));
        size = Math.Max(size, (int)Math.Round(box.H * sy));
        // Largest square that can fully fit inside the image.
        size = Math.Min(size, Math.Min(srcW, srcH));
        int cx = (int)Math.Round(box.Cx * sx);
        int cy = (int)Math.Round(box.Cy * sy);
        int x = Math.Clamp(cx - size / 2, 0, srcW - size);
        int y = Math.Clamp(cy - size / 2, 0, srcH - size);
        return (x, y, size);
    }

    /// <summary>
    /// Produces a 1024x1024 region centered on a concept box, clamped to the base
    /// image bounds.
    /// </summary>
    public static CropRegion CenteredOn(ConceptBox box, int baseW, int baseH)
    {
        int x = (int)Math.Round(box.Cx - CropSize / 2f);
        int y = (int)Math.Round(box.Cy - CropSize / 2f);
        x = Math.Clamp(x, 0, Math.Max(0, baseW - CropSize));
        y = Math.Clamp(y, 0, Math.Max(0, baseH - CropSize));
        return new CropRegion
        {
            X = x,
            Y = y,
            Width = Math.Min(CropSize, baseW),
            Height = Math.Min(CropSize, baseH),
            Concept = box.Concept,
            IsConceptCentered = true,
            SourceBox = box
        };
    }
}
