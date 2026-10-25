using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DatasetCropper.Ml;

/// <summary>Generates natural-language text for an image crop.</summary>
public interface ICaptioner : IDisposable
{
    string Name { get; }
    string Caption(Image<Rgba32> image, double threshold);
}
