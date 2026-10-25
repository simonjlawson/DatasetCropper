using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DatasetCropper.Imaging;

/// <summary>
/// Converts output crops (PNG) to JPG in place. <c>*-masklabel.png</c> files are
/// left alone — masks stay grayscale PNG.
/// </summary>
public static class JpgConversion
{
    public const int GoodQuality = 75;
    public const int BetterQuality = 85;
    public const int JpegQuality = 95;

    public static int QualityForLevel(int level) => level switch
    {
        0 => GoodQuality,
        1 => BetterQuality,
        2 => JpegQuality,
        _ => throw new ArgumentOutOfRangeException(nameof(level), level, "Quality level must be between 0 and 2.")
    };

    /// <summary>
    /// Converts every <c>*.png</c> under <paramref name="root"/> (hidden folders
    /// skipped) that is not a <c>*-masklabel.png</c> to a same-stem <c>*.jpg</c>,
    /// deleting the PNG on success. An existing <c>*.jpg</c> of the same stem is
    /// never overwritten. Returns (converted, skipped, failed) — a failure leaves
    /// its PNG in place.
    /// </summary>
    public static (int Converted, int Skipped, int Failed) ConvertPngs(string root, int quality = JpegQuality)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(quality, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(quality, 100);

        int converted = 0, skipped = 0, failed = 0;
        var dirs = new Queue<string>();
        dirs.Enqueue(root);
        while (dirs.Count > 0)
        {
            string dir = dirs.Dequeue();

            foreach (string sub in Safe(Directory.EnumerateDirectories(dir)))
            {
                string name = Path.GetFileName(sub) ?? "";
                if (name.Length > 0 && name[0] != '.')
                    dirs.Enqueue(sub);
            }

            foreach (string png in Safe(Directory.EnumerateFiles(dir, "*.png")))
            {
                if (png.EndsWith("-masklabel.png", StringComparison.OrdinalIgnoreCase))
                    continue;

                string jpg = Path.ChangeExtension(png, ".jpg");
                if (File.Exists(jpg)) { skipped++; continue; }

                try
                {
                    using var img = Image.Load<Rgba32>(png);
                    // JPG has no alpha channel: flatten transparency onto white first.
                    img.Mutate(x => x.BackgroundColor(Color.White));
                    img.SaveAsJpeg(jpg, new JpegEncoder { Quality = quality });
                    File.Delete(png);
                    converted++;
                }
                catch { failed++; } // locked or corrupt — the PNG stays in place
            }
        }
        return (converted, skipped, failed);
    }

    /// <summary>Materializes a lazy enumeration; access-denied folders just contribute nothing.</summary>
    private static List<string> Safe(IEnumerable<string> files)
    {
        try { return files.ToList(); }
        catch { return new List<string>(); }
    }
}
