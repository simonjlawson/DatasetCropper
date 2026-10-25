namespace DatasetCropper.Imaging;

/// <summary>
/// Filters a batch of file paths down to the image files that should be
/// queued/processed. Centralized so the WPF queue and the CLI behave
/// identically.
/// </summary>
public static class ImageFileFilter
{
    /// <summary>Image extensions accepted by the pipeline (lowercase, with dot).</summary>
    public static readonly string[] ImageExtensions =
        { ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff" };

    /// <summary>True if the path has a known image extension (case-insensitive).</summary>
    public static bool IsImage(string path) =>
        ImageExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>
    /// Flattens <paramref name="paths"/>: any directory entry is expanded
    /// recursively into the files it contains, other entries pass through
    /// unchanged. Result is ordered by path so the queue order is stable.
    /// </summary>
    public static IEnumerable<string> ExpandDirectories(IEnumerable<string> paths)
        => paths.SelectMany(p => Directory.Exists(p)
            ? Directory.EnumerateFiles(p, "*", SearchOption.AllDirectories)
            : new[] { p })
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Filters <paramref name="files"/> to image files, then drops any
    /// .png whose stem collides with another image file in the same batch
    /// (the non-png original is preferred). Returns a new list; order is
    /// preserved.
    /// </summary>
    public static List<string> FilterFiles(IEnumerable<string> files)
    {
        var images = files.Where(IsImage).ToList();
        // Stems present as a NON-png image in this batch.
        var nonPngStems = images
            .Where(f => !Path.GetExtension(f).Equals(".png", StringComparison.OrdinalIgnoreCase))
            .Select(f => Path.GetFileNameWithoutExtension(f))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        return images
            .Where(f => !(Path.GetExtension(f).Equals(".png", StringComparison.OrdinalIgnoreCase)
                          && nonPngStems.Contains(Path.GetFileNameWithoutExtension(f))))
            .ToList();
    }
}
