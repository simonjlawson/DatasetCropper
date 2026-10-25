using DatasetCropper.Imaging;
using DatasetCropper.Models;
using Xunit;

namespace DatasetCropper.Tests;

public class CropTilerTests
{
    [Theory]
    [InlineData(1024, 1024)]
    [InlineData(2048, 1024)]
    [InlineData(1024, 2048)]
    [InlineData(3000, 2000)]
    [InlineData(1500, 1500)]
    [InlineData(1024, 1536)]
    [InlineData(1023, 1023)]
    public void BaseSize_SmallestDimensionBecomes1024(int w, int h)
    {
        var (bw, bh) = CropTiler.BaseSize(w, h);
        Assert.Equal(1024, Math.Min(bw, bh));
    }

    [Theory]
    [InlineData(2048, 1024)]
    [InlineData(1024, 2048)]
    [InlineData(512, 512)]
    public void BaseSize_PreservesAspectRatio(int w, int h)
    {
        var (bw, bh) = CropTiler.BaseSize(w, h);
        // Aspect ratio is preserved within rounding (1px) tolerance.
        Assert.True(Math.Abs((double)bw / bh - (double)w / h) < 0.01);
    }

    [Theory]
    [InlineData(1024, 1024)]
    [InlineData(2048, 1024)]
    [InlineData(1024, 2048)]
    [InlineData(3000, 2000)]
    [InlineData(1500, 1500)]
    [InlineData(1024, 1536)]
    [InlineData(1536, 1024)]
    public void Tile_CoversEntireBaseImage_NoGaps_NoOverrun(int baseW, int baseH)
    {
        var tiles = CropTiler.Tile(baseW, baseH);
        Assert.NotEmpty(tiles);

        var xCovered = new bool[baseW];
        var yCovered = new bool[baseH];

        foreach (var t in tiles)
        {
            // Every tile must sit fully inside the base image.
            Assert.True(t.X >= 0);
            Assert.True(t.Y >= 0);
            Assert.True(t.X + t.Width <= baseW);
            Assert.True(t.Y + t.Height <= baseH);

            for (int x = t.X; x < t.X + t.Width; x++) xCovered[x] = true;
            for (int y = t.Y; y < t.Y + t.Height; y++) yCovered[y] = true;
        }

        // Full coverage: no gaps along either axis.
        Assert.All(xCovered, c => Assert.True(c));
        Assert.All(yCovered, c => Assert.True(c));
    }

    [Fact]
    public void Tile_SingleTileWhenBaseIsExactlyOneCrop()
    {
        var tiles = CropTiler.Tile(1024, 1024);
        Assert.Single(tiles);
        var t = tiles[0];
        Assert.Equal(0, t.X);
        Assert.Equal(0, t.Y);
        Assert.Equal(1024, t.Width);
        Assert.Equal(1024, t.Height);
    }

    [Theory]
    [InlineData(2048, 2048, 1024, 1024)] // dead center
    [InlineData(2048, 2048, 100, 100)]   // near top-left
    [InlineData(2048, 2048, 2000, 2000)] // near bottom-right
    [InlineData(1024, 1024, 512, 512)]   // exact center of a 1-crop base
    public void CenteredOn_RegionIsWithinBounds(int baseW, int baseH, float cx, float cy)
    {
        var box = new ConceptBox { Concept = "cat", X = cx - 50, Y = cy - 50, W = 100, H = 100, Score = 1f };
        var region = CropTiler.CenteredOn(box, baseW, baseH);

        Assert.True(region.X >= 0);
        Assert.True(region.Y >= 0);
        Assert.True(region.X + region.Width <= baseW);
        Assert.True(region.Y + region.Height <= baseH);
        Assert.Equal(Math.Min(CropTiler.CropSize, baseW), region.Width);
        Assert.Equal(Math.Min(CropTiler.CropSize, baseH), region.Height);
        Assert.True(region.IsConceptCentered);
        Assert.Equal("cat", region.Concept);
        Assert.Same(box, region.SourceBox);
    }

    [Fact]
    public void CenteredOn_CentersOnBoxWhenItFits()
    {
        // Box centered at (1024,1024) in a 2048x2048 base.
        var box = new ConceptBox { Concept = "cat", X = 974, Y = 974, W = 100, H = 100, Score = 1f };
        var region = CropTiler.CenteredOn(box, 2048, 2048);

        Assert.Equal(box.Cx, region.X + region.Width / 2f, 1);
        Assert.Equal(box.Cy, region.Y + region.Height / 2f, 1);
    }

    [Fact]
    public void StepForOverlap_DefaultOverlap_Is512()
    {
        Assert.Equal(512, CropTiler.StepForOverlap(512));
    }

    [Fact]
    public void StepForOverlap_Derivation()
    {
        Assert.Equal(256, CropTiler.StepForOverlap(768));
        Assert.Equal(768, CropTiler.StepForOverlap(256));
    }

    [Fact]
    public void StepForOverlap_Clamped()
    {
        Assert.Equal(CropTiler.CropSize, CropTiler.StepForOverlap(0));
        Assert.Equal(CropTiler.CropSize, CropTiler.StepForOverlap(-10));
        Assert.Equal(1, CropTiler.StepForOverlap(2048));
    }

    [Fact]
    public void Tile_MoreOverlap_MoreTiles()
    {
        Assert.Equal(3, CropTiler.Tile(2048, 1024, 512).Count);
        Assert.True(CropTiler.Tile(2048, 1024, 768).Count > CropTiler.Tile(2048, 1024, 512).Count);
    }

    [Fact]
    public void Tile_LessOverlap_FewerTiles()
    {
        Assert.True(CropTiler.Tile(4096, 1024, 256).Count < CropTiler.Tile(4096, 1024, 512).Count);
    }

    [Fact]
    public void Tile_CustomOverlap_StillCoversEntireBase()
    {
        int baseW = 2048, baseH = 1024;
        var tiles = CropTiler.Tile(baseW, baseH, 768)
            .OrderBy(t => t.X).ThenBy(t => t.Y).ToList();

        Assert.NotEmpty(tiles);
        Assert.Equal(0, tiles.First().X);
        Assert.Contains(tiles, t => t.X + t.Width >= baseW);

        // No gaps: each tile starts before the previous one ends.
        int prevEnd = 0;
        foreach (var t in tiles)
        {
            Assert.True(t.X <= prevEnd, $"Gap between {prevEnd} and {t.X}");
            prevEnd = Math.Max(prevEnd, t.X + t.Width);
        }
        Assert.True(prevEnd >= baseW);
    }

    [Theory]
    [InlineData(1024, 1024, 1024, 1024, 462, 462, 100, 100, 0, 0, 1024)] // small box, 1:1 scale -> 1024 square
    [InlineData(1024, 1024, 1024, 1024, 0, 0, 50, 50, 0, 0, 1024)] // corner box -> window slides to the edge, no off-edge crop
    [InlineData(4096, 4096, 8192, 8192, 1048, 1048, 2000, 2000, 2096, 2096, 4000)] // big box at 2x scale -> box-sized square
    [InlineData(2048, 1024, 4096, 2048, 1024, 462, 100, 100, 1636, 512, 1024)] // 2x scale both axes
    [InlineData(2048, 4096, 2048, 4096, 974, 512, 100, 3000, 0, 988, 2048)] // tall box -> size capped at image width, clamped inside
    [InlineData(2048, 4096, 2048, 4096, 0, 0, 2048, 4096, 0, 1024, 2048)] // full-image box -> largest in-bounds square, centered
    public void SubjectSquare_CentersOnBoxClampedInsideImage(
        int baseW, int baseH, int srcW, int srcH,
        float x, float y, float w, float h,
        int expectedX, int expectedY, int expectedSize)
    {
        var box = new ConceptBox { Concept = "face", X = x, Y = y, W = w, H = h, Score = 1f };
        var (ex, ey, size) = CropTiler.SubjectSquare(box, baseW, baseH, srcW, srcH);

        Assert.Equal(expectedSize, size);
        Assert.Equal(expectedX, ex);
        Assert.Equal(expectedY, ey);

        // The square always fits fully inside the image (no padded black space).
        Assert.True(size >= 1 && size <= srcW && size <= srcH);
        Assert.True(ex >= 0 && ex + size <= srcW);
        Assert.True(ey >= 0 && ey + size <= srcH);

        // Centered on the box's full-resolution center when that keeps the
        // square in-bounds, otherwise slid to the nearest image edge.
        double sx = (double)srcW / baseW, sy = (double)srcH / baseH;
        double idealX = box.Cx * sx - size / 2d, idealY = box.Cy * sy - size / 2d;
        Assert.True(Math.Abs((int)Math.Clamp(idealX, 0, srcW - size) - ex) <= 1);
        Assert.True(Math.Abs((int)Math.Clamp(idealY, 0, srcH - size) - ey) <= 1);
    }

    [Fact]
    public void SubjectSquare_UsesConfiguredMinimumResolution()
    {
        var box = new ConceptBox { Concept = "face", X = 462, Y = 462, W = 100, H = 100, Score = 1f };

        var (_, _, size) = CropTiler.SubjectSquare(box, 1024, 1024, 4096, 4096, 1536);

        Assert.Equal(1536, size);
    }
}
