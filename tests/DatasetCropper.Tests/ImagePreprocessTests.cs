using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Ml;
using Xunit;

namespace DatasetCropper.Tests;

public class ImagePreprocessTests
{
    // ImageNet normalization constants (as used by DINO / SAM).
    private const float MeanR = 0.485f, MeanG = 0.456f, MeanB = 0.406f;
    private const float StdR = 0.229f, StdG = 0.224f, StdB = 0.225f;

    [Fact]
    public void ToChwTensor_UsesChwLayout()
    {
        // 2x2 image. Distinct colors per pixel so planes are distinguishable:
        //   (0,0)=(10,20,30)  (1,0)=(40,50,60)
        //   (0,1)=(70,80,90)  (1,1)=(100,110,120)
        byte[] rgb =
        {
            10, 20, 30,   40, 50, 60,
            70, 80, 90,   100, 110, 120
        };
        var t = ImagePreprocess.ToChwTensor(rgb, 2, 2);
        Assert.Equal(12, t.Length); // 3 * 2 * 2

        int plane = 4;
        // R plane = R channel of each pixel, row-major.
        Assert.Equal(10, t[0]);
        Assert.Equal(40, t[1]);
        Assert.Equal(70, t[2]);
        Assert.Equal(100, t[3]);
        // G plane.
        Assert.Equal(20, t[plane + 0]);
        Assert.Equal(50, t[plane + 1]);
        Assert.Equal(80, t[plane + 2]);
        Assert.Equal(110, t[plane + 3]);
        // B plane.
        Assert.Equal(30, t[2 * plane + 0]);
        Assert.Equal(60, t[2 * plane + 1]);
        Assert.Equal(90, t[2 * plane + 2]);
        Assert.Equal(120, t[2 * plane + 3]);
    }

    [Fact]
    public void ToChwTensor_RescaleOnly_NoNormalization()
    {
        byte[] rgb = { 128, 64, 0 }; // 1x1
        var t = ImagePreprocess.ToChwTensor(rgb, 1, 1, rescale: 1f / 255f);
        Assert.Equal(128f / 255f, t[0], 6);
        Assert.Equal(64f / 255f, t[1], 6);
        Assert.Equal(0f, t[2], 6);
    }

    [Fact]
    public void ToChwTensor_ImageNetNormalization_MatchesHandComputed()
    {
        // 1x1 image, R=100 G=150 B=200, /255 then (v-mean)/std.
        byte[] rgb = { 100, 150, 200 };
        var t = ImagePreprocess.ToChwTensor(rgb, 1, 1,
            rescale: 1f / 255f,
            meanR: MeanR, meanG: MeanG, meanB: MeanB,
            stdR: StdR, stdG: StdG, stdB: StdB);

        float r = 100f / 255f;
        float g = 150f / 255f;
        float b = 200f / 255f;
        Assert.Equal((r - MeanR) / StdR, t[0], 5);
        Assert.Equal((g - MeanG) / StdG, t[1], 5);
        Assert.Equal((b - MeanB) / StdB, t[2], 5);

        // Spot-check the actual magnitudes (guards against a silently wrong
        // constant set, not just internal consistency).
        Assert.InRange(t[0], -0.5f, -0.3f);   // R below mean -> negative
        Assert.InRange(t[1], 0.5f, 0.7f);     // G above mean -> positive
        Assert.InRange(t[2], 1.5f, 1.9f);     // B well above mean
    }

    [Fact]
    public void ToChwTensor_NoNormalizationWhenMeanStdNull()
    {
        byte[] rgb = { 100, 150, 200 };
        var t = ImagePreprocess.ToChwTensor(rgb, 1, 1, rescale: 1f / 255f);
        Assert.Equal(100f / 255f, t[0], 6);
        Assert.Equal(150f / 255f, t[1], 6);
        Assert.Equal(200f / 255f, t[2], 6);
    }

    // ---------- PaddedCrop ----------

    private static void FillGradient(Image<Rgba32> img)
    {
        for (int y = 0; y < img.Height; y++)
            for (int x = 0; x < img.Width; x++)
                img[x, y] = new Rgba32((byte)(x % 251), (byte)(y % 251), 50, 255);
    }

    [Fact]
    public void PaddedCrop_RegionFullyInside_MatchesCropAndResize()
    {
        // 200x200 source, a 50x50 window fully inside, upscaled 2x.
        var src = new Image<Rgba32>(200, 200);
        FillGradient(src);

        using var a = ImagePreprocess.PaddedCrop(src, 50, 50, 50, 50, 100, 100, (0, 0, 0));
        using var b = ImagePreprocess.CropAndResize(src, 50, 50, 50, 50, 100, 100);
        Assert.Equal(ImagePreprocess.ToRgbBytes(b, (0, 0, 0)), ImagePreprocess.ToRgbBytes(a, (0, 0, 0)));
    }

    [Fact]
    public void PaddedCrop_RegionOverflowingLeft_PadsAndPreservesAspectRatio()
    {
        // 100x100 source (R = x gradient), region starts at x = -50: half the
        // canvas must be padding, and the in-bounds half must be the left half
        // of the source at 2x scale — NOT the whole source crushed into a half.
        var src = new Image<Rgba32>(100, 100);
        for (int x = 0; x < 100; x++) for (int y = 0; y < 100; y++) src[x, y] = new Rgba32((byte)x, 0, 0, 255);

        using var canvas = ImagePreprocess.PaddedCrop(src, -50, 0, 100, 100, 200, 200, (0, 0, 0));
        Assert.Equal(200, canvas.Width);

        // Left half is padding, right half is source slice.
        Assert.Equal(0, canvas[20, 100].R);          // padding
        Assert.Equal(0, canvas[99, 100].R);          // padding
        Assert.Equal(1, canvas[102, 100].R);         // slice dst 2  -> src 1
        Assert.Equal(49, canvas[100 + 98, 100].R);   // slice dst 98 -> src 49
        Assert.Equal(49, canvas[199, 100].R);        // dst 99  -> src 49
    }

    [Fact]
    public void PaddedCrop_RegionFullyOutside_ReturnsFilledCanvas()
    {
        using var src = ImagePreprocess.NewFilled(10, 10, 10, 20, 30);
        using var canvas = ImagePreprocess.PaddedCrop(src, 500, 500, 100, 100, 50, 50, (255, 255, 255));
        Assert.Equal(50, canvas.Width);
        Assert.Equal(50, canvas.Height);
        for (int y = 0; y < 50; y++)
            for (int x = 0; x < 50; x++)
                Assert.Equal(new Rgba32(255, 255, 255), canvas[x, y]);
    }

    [Fact]
    public void PaddedCrop_RealisticSmallImage_SqueezesNothing()
    {
        // Regression: an 800x600 image with a >=1024 square centered near its
        // center used to clamp+stretch the *whole* image into the crop.
        // Now the 800x600 image occupies exactly (112, 212)-(911, 811) and
        // everything outside is black padding.
        var src = new Image<Rgba32>(800, 600);
        for (int y = 0; y < 600; y++)
            for (int x = 0; x < 800; x++)
                src[x, y] = new Rgba32((byte)(x % 251), (byte)((y * 7) % 251), 50, 255);

        using var crop = ImagePreprocess.PaddedCrop(src, -112, -212, 1024, 1024, 1024, 1024, (0, 0, 0));
        Assert.Equal(1024, crop.Width);

        // Corners / edges: padding.
        Assert.Equal(0, crop[0, 0].R);
        Assert.Equal(0, crop[50, 100].R);
        Assert.Equal(0, crop[920, 300].R);          // x=912+ is right padding
        Assert.Equal(0, crop[200, 900].R);          // y=812+ is bottom padding

        // In-bounds corners sample the source corners at 1:1 (no squeeze):
        Assert.Equal(0, crop[112, 212].R);          // src (0, 0)
        Assert.Equal(0, crop[112, 212].G);
        Assert.Equal(46, crop[911, 811].R);         // src (799, 599): 799%251
        Assert.Equal(177, crop[911, 811].G);        // (599*7)%251
    }

    // ---------- SamEncoding.CanvasGeometry ----------

    [Fact]
    public void CanvasGeometry_SquareImage_NoPad()
    {
        var (scale, padX, padY) = DatasetCropper.Ml.SamEncoding.CanvasGeometry(1024, 1024);
        Assert.Equal(1f, scale, 6);
        Assert.Equal(0, padX);
        Assert.Equal(0, padY);
    }

    [Fact]
    public void CanvasGeometry_NonSquare_CentersWithExpectedPad()
    {
        // 2048x1024: scale 0.5, image 1024x512, centered -> padY 256.
        var (s1, px1, py1) = DatasetCropper.Ml.SamEncoding.CanvasGeometry(2048, 1024);
        Assert.Equal(0.5f, s1, 6);
        Assert.Equal(0, px1);
        Assert.Equal(256, py1);

        // 683x1024 (a real 16:10 shape): scale 1.0, 683x1024 -> padX 170.
        var (s2, px2, py2) = DatasetCropper.Ml.SamEncoding.CanvasGeometry(683, 1024);
        Assert.Equal(1f, s2, 6);
        Assert.Equal(170, px2);
        Assert.Equal(0, py2);
    }
}
