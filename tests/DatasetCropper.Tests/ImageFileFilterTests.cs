using System.IO;
using System.Linq;
using DatasetCropper.Imaging;
using Xunit;

namespace DatasetCropper.Tests;

public class ImageFileFilterTests
{
    [Theory]
    [InlineData("a.png")]
    [InlineData("a.jpg")]
    [InlineData("a.jpeg")]
    [InlineData("a.bmp")]
    [InlineData("a.webp")]
    [InlineData("a.tiff")]
    [InlineData("PHO.JPG")] // mixed case
    public void IsImage_KnownExtensions_ReturnsTrue(string path)
    {
        Assert.True(ImageFileFilter.IsImage(path));
    }

    [Theory]
    [InlineData("a.txt")]
    [InlineData("a.gif")]
    [InlineData("a")] // no extension
    [InlineData(@"C:\pics")] // directory-like path
    public void IsImage_UnknownExtensions_ReturnsFalse(string path)
    {
        Assert.False(ImageFileFilter.IsImage(path));
    }

    [Fact]
    public void FilterFiles_PngCollidesWithSameStemImage_DropsPng()
    {
        var result = ImageFileFilter.FilterFiles(new[] { "a.png", "a.jpg" });

        Assert.Equal(new[] { "a.jpg" }, result);
    }

    [Fact]
    public void FilterFiles_PngAlone_Survives()
    {
        var result = ImageFileFilter.FilterFiles(new[] { "a.png" });

        Assert.Equal(new[] { "a.png" }, result);
    }

    [Fact]
    public void FilterFiles_PngWithSameStemTxt_KeepsPng()
    {
        var result = ImageFileFilter.FilterFiles(new[] { "a.png", "a.txt" });

        Assert.Equal(new[] { "a.png" }, result);
    }

    [Fact]
    public void FilterFiles_DifferentStems_BothSurvive()
    {
        var result = ImageFileFilter.FilterFiles(new[] { "x.jpg", "y.png" });

        Assert.Equal(new[] { "x.jpg", "y.png" }, result);
    }

    [Fact]
    public void FilterFiles_PreservesOrder()
    {
        var result = ImageFileFilter.FilterFiles(new[]
        {
            "b.png", "a.png", "b.jpg", "c.txt", "a.jpg", "d.png",
        });

        Assert.Equal(new[] { "b.jpg", "a.jpg", "d.png" }, result);
    }

    [Fact]
    public void FilterFiles_ExtensionComparisonIsCaseInsensitive()
    {
        // "A.PNG" is a png even though the extension is uppercase; it still
        // collides with "a.jpg" and is dropped.
        var result = ImageFileFilter.FilterFiles(new[] { "A.PNG", "a.jpg" });

        Assert.Equal(new[] { "a.jpg" }, result);
    }

    [Fact]
    public void ExpandDirectories_NonDirectory_PassesThrough()
    {
        var result = ImageFileFilter.ExpandDirectories(new[] { "a.png" }).ToList();

        Assert.Equal(new[] { "a.png" }, result);
    }

    [Fact]
    public void ExpandDirectories_Filtered_FolderYieldsOnlyImagesRecursively()
    {
        var root = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(Path.Combine(root, "sub"));
        try
        {
            File.WriteAllText(Path.Combine(root, "b.png"), "");
            File.WriteAllText(Path.Combine(root, "notes.txt"), "");
            File.WriteAllText(Path.Combine(root, "sub", "a.jpg"), "");
            File.WriteAllText(Path.Combine(root, "sub", "c.bmp"), "");

            var result = ImageFileFilter
                .FilterFiles(ImageFileFilter.ExpandDirectories(new[] { root }))
                .Select(f => Path.GetFileName(f))
                .ToList();

            // Non-images excluded; order is by full path (sub/ sorts after root files).
            Assert.Equal(new[] { "b.png", "a.jpg", "c.bmp" }, result);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
