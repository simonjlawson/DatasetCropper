using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Imaging;
using Xunit;

namespace DatasetCropper.Tests;

public sealed class MaskUtilsTests
{
    [Fact]
    public void ToBinaryMask_EmitsOnlyBlackAndWhitePixels()
    {
        using var source = new Image<Rgba32>(3, 1);
        source[0, 0] = new Rgba32(0, 0, 0);
        source[1, 0] = new Rgba32(127, 127, 127);
        source[2, 0] = new Rgba32(255, 255, 255);

        using var result = MaskUtils.ToBinaryMask(source, 128);

        Assert.Equal(0, result[0, 0].R);
        Assert.Equal(0, result[1, 0].R);
        Assert.Equal(255, result[2, 0].R);
        for (int x = 0; x < result.Width; x++)
        {
            Assert.Equal(result[x, 0].R, result[x, 0].G);
            Assert.Equal(result[x, 0].R, result[x, 0].B);
        }
    }
}
