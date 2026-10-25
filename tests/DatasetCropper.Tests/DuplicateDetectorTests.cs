using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Imaging;
using Xunit;

namespace DatasetCropper.Tests;

public class DuplicateDetectorTests
{
    /// <summary>Square image with a horizontal brightness ramp (optionally inverted).</summary>
    private static Image<Rgba32> Ramp(bool inverted = false, int size = 256)
    {
        var image = new Image<Rgba32>(size, size);
        image.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = acc.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    int v = (int)(255.0 * x / (size - 1));
                    if (inverted) v = 255 - v;
                    row[x] = new Rgba32((byte)v, (byte)v, (byte)v, 255);
                }
            }
        });
        return image;
    }

    /// <summary>Square image painted a single flat color.</summary>
    private static Image<Rgba32> Flat(byte v = 128, int size = 64)
    {
        var image = new Image<Rgba32>(size, size);
        image.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = acc.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                    row[x] = new Rgba32(v, v, v, 255);
            }
        });
        return image;
    }

    [Fact]
    public void DHash_SameContentDifferentPixelSize_Matches()
    {
        using var a = Ramp(false, 128);
        using var b = Ramp(false, 512);
        Assert.True(
            DuplicateDetector.Distance(DuplicateDetector.DHash(a), DuplicateDetector.DHash(b)) <= 2,
            "re-samples of identical content must hash (nearly) identically");
    }

    [Fact]
    public void DHash_InvertedContent_Differs()
    {
        using var a = Ramp(false);
        using var b = Ramp(true);
        Assert.True(
            DuplicateDetector.Distance(DuplicateDetector.DHash(a), DuplicateDetector.DHash(b)) > 40,
            "a left-to-right and a right-to-left ramp invert nearly every comparison bit");
    }

    [Fact]
    public void DHash_FlatImage_IsZero()
    {
        using var flat = Flat();
        Assert.Equal(0UL, DuplicateDetector.DHash(flat));
    }

    [Fact]
    public void Distance_ZeroForEqualHashes()
    {
        using var a = Ramp();
        using var b = Ramp();
        ulong h1 = DuplicateDetector.DHash(a);
        Assert.Equal(0, DuplicateDetector.Distance(h1, h1));
    }

    [Fact]
    public void Tracker_FlagsNearDuplicate_NotDifferentContent()
    {
        var tracker = new DedupeTracker(8);

        using var first = Ramp(false, 128);
        using var close = Ramp(false, 192); // same content, other pixel size
        using var different = Ramp(true, 128);

        ulong h1 = DuplicateDetector.DHash(first);
        Assert.False(tracker.IsDuplicate(h1, "img_000.png", out _),
            "the first crop has no reference to duplicate yet");
        tracker.Remember(h1, "img_000.png");

        Assert.True(tracker.IsDuplicate(DuplicateDetector.DHash(close), "img_001.png", out string? dupOf));
        Assert.Equal("img_000.png", dupOf);

        Assert.False(tracker.IsDuplicate(DuplicateDetector.DHash(different), "img_002.png", out _),
            "an inverted ramp is different content, not a duplicate");
    }
}
