namespace DatasetCropper.Models;

/// <summary>
/// A rectangular region (in the base image's pixel coordinates) that will be
/// extracted as a 1024x1024 training crop.
/// </summary>
public sealed class CropRegion
{
    public int X { get; init; }
    public int Y { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }

    /// <summary>The concept this crop is assigned to (folder name), or null when unassigned (the image goes to the "unsorted" folder).</summary>
    public string? Concept { get; init; }

    /// <summary>True when this crop was generated to center on a concept box (vs. a tile).</summary>
    public bool IsConceptCentered { get; init; }

    /// <summary>The box the crop is centered on, when IsConceptCentered.</summary>
    public ConceptBox? SourceBox { get; init; }

    /// <summary>
    /// True for subject crops (second concept list): the crop is cut from the
    /// full-resolution original at a 1:1 square of at least 1024x1024, and is
    /// emitted into a folder named after its own concept (not the image's
    /// dominant concept).
    /// </summary>
    public bool IsSubject { get; init; }

    public override string ToString() =>
        $"({X},{Y} {Width}x{Height}) -> {Concept ?? "unsorted"}{(IsConceptCentered ? " [centered]" : "")}";
}
