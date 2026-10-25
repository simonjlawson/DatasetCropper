namespace DatasetCropper.Models;

/// <summary>
/// A detected concept instance: the matched concept text plus its bounding box in
/// the base (resized) image's pixel coordinates.
/// </summary>
public sealed class ConceptBox
{
    public required string Concept { get; init; }
    public float X { get; init; }   // left
    public float Y { get; init; }   // top
    public float W { get; init; }   // width
    public float H { get; init; }   // height
    public float Score { get; init; }

    public float X2 => X + W;
    public float Y2 => Y + H;
    public float Cx => X + W / 2f;
    public float Cy => Y + H / 2f;

    public override string ToString() =>
        $"{Concept} [{X:0.#},{Y:0.#} {W:0.#}x{H:0.#}] ({Score:0.##})";
}
