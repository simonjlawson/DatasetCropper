namespace DatasetCropper.Models;

public enum ImageStatus
{
    Queued,
    Upscaling,
    Detecting,
    Cropping,
    Masking,
    Tagging,
    Done,
    Error,
    Skipped
}

/// <summary>Immutable snapshot of overall processing progress, pushed to the UI.</summary>
public sealed record ProgressReport(
    int Total,
    int Completed,
    string? CurrentImage,
    string? Stage,
    double OverallFraction,
    string? Message)
{
}
