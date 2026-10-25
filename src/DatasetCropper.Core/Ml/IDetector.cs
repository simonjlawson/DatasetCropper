using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// A concept detector (Grounding DINO, YOLO, …). The pipeline always calls
/// <see cref="Detect"/> with an arbitrary list of concept strings and receives
/// detections in the image's own pixel coordinates. Implementations load their
/// ONNX session once (see each type's <c>TryLoad</c>) and are then shared across
/// the queue under the caller's model lock — so they must be safe to call
/// re-entrantly but not concurrently.
/// </summary>
public interface IDetector : IDisposable
{
    /// <summary>Display name ("Grounding DINO", "YOLO").</summary>
    string Name { get; }

    /// <summary>True once the model is loaded and usable.</summary>
    bool IsLoaded { get; }

    /// <summary>
    /// Detects <paramref name="concepts"/> in <paramref name="image"/> and returns
    /// one <see cref="ConceptBox"/> per detection (in the image's pixel coordinates),
    /// each tagged with the matched concept.
    /// <paramref name="boxThreshold"/> is the box/score confidence gate (0..1).
    /// <paramref name="textThreshold"/> is the text-token gate Grounding DINO uses;
    /// detectors that don't score text tokens (YOLO) ignore it.
    /// </summary>
    List<ConceptBox> Detect(Image<Rgba32> image, IReadOnlyList<string> concepts,
        double boxThreshold, double textThreshold);
}
