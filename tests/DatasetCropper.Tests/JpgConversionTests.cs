using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DatasetCropper.Imaging;
using Xunit;

namespace DatasetCropper.Tests;

public class JpgConversionTests
{
    [Theory]
    [InlineData(0, 75)]
    [InlineData(1, 85)]
    [InlineData(2, 95)]
    public void QualityForLevel_ReturnsExpectedJpegQuality(int level, int expected)
    {
        Assert.Equal(expected, JpgConversion.QualityForLevel(level));
    }

    [Fact]
    public void QualityForLevel_InvalidLevel_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => JpgConversion.QualityForLevel(3));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(101)]
    public void ConvertPngs_InvalidQuality_Throws(int quality)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => JpgConversion.ConvertPngs("unused", quality));
    }

    private static void WritePng(string path)
    {
        using var img = new Image<Rgba32>(16, 16);
        img.SaveAsPng(path);
    }

    [Fact]
    public void ConvertPngs_CropsBecomeJpg_MasksStayPng()
    {
        string root = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(Path.Combine(root, "cat"));
        try
        {
            WritePng(Path.Combine(root, "cat", "cat001.png"));
            WritePng(Path.Combine(root, "cat", "cat001-masklabel.png"));

            (int converted, int skipped, int failed) = JpgConversion.ConvertPngs(root);

            Assert.Equal(1, converted);
            Assert.Equal(0, skipped);
            Assert.Equal(0, failed);
            Assert.True(File.Exists(Path.Combine(root, "cat", "cat001.jpg")));
            Assert.False(File.Exists(Path.Combine(root, "cat", "cat001.png")));
            // The mask is untouched: still a PNG, still loadable.
            string mask = Path.Combine(root, "cat", "cat001-masklabel.png");
            Assert.True(File.Exists(mask));
            using (var maskImg = Image.Load(mask))
            {
                Assert.True(maskImg.Frames.Count > 0);
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ConvertPngs_SkipsHiddenFolders_AndRootLevelPngsAreConverted()
    {
        string root = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(Path.Combine(root, ".topaz", "work"));
        Directory.CreateDirectory(Path.Combine(root, "dog"));
        try
        {
            WritePng(Path.Combine(root, "dog", "d001.png"));
            WritePng(Path.Combine(root, "top.png"));           // root-level crop
            WritePng(Path.Combine(root, ".topaz", "hidden.png")); // must NOT be touched

            (int converted, int skipped, int failed) = JpgConversion.ConvertPngs(root);

            Assert.Equal(2, converted);
            Assert.True(File.Exists(Path.Combine(root, "dog", "d001.jpg")));
            Assert.True(File.Exists(Path.Combine(root, "top.jpg")));
            Assert.True(File.Exists(Path.Combine(root, ".topaz", "hidden.png")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ConvertPngs_ExistingJpg_IsNotOverwritten()
    {
        string root = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(root);
        try
        {
            WritePng(Path.Combine(root, "a.png"));
            WritePng(Path.Combine(root, "a.jpg"));
            long existingSize = new FileInfo(Path.Combine(root, "a.jpg")).Length;

            (int converted, int skipped, int failed) = JpgConversion.ConvertPngs(root);

            Assert.Equal(0, converted);
            Assert.Equal(1, skipped);
            Assert.Equal(0, failed);
            // Originals untouched: PNG left in place, JPG unmodified.
            Assert.True(File.Exists(Path.Combine(root, "a.png")));
            Assert.Equal(existingSize, new FileInfo(Path.Combine(root, "a.jpg")).Length);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
