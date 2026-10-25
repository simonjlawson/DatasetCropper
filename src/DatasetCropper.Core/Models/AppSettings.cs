using System.IO;

namespace DatasetCropper.Models;

/// <summary>
/// Persisted user settings (concepts, paths, thresholds, Topaz options).
/// </summary>
public sealed class AppSettings
{
    /// <summary>Predefined concepts, one per line (order preserved).</summary>
    public List<string> Concepts { get; set; } = new() { "cat", "dog" };

    /// <summary>
    /// Second concept list: subject concepts searched for in the original
    /// (full-resolution) image. Each match is emitted as a 1:1 crop of at least
    /// 1024x1024 into its own &lt;OutputFolder&gt;/&lt;subject&gt;/ folder — separate
    /// from the composition-concept crops. Empty by default (feature off).
    /// </summary>
    public List<string> SubjectConcepts { get; set; } = new();

    /// <summary>Optional terms whose detected regions are used for every emitted mask.</summary>
    public List<string> MaskTerms { get; set; } = new();

    /// <summary>Output root folder. Crops are written to &lt;OutputFolder&gt;/&lt;concept&gt;/.</summary>
    public string OutputFolder { get; set; } = "output";

    /// <summary>Append newly generated text below an existing caption file.</summary>
    public bool AppendCaptions { get; set; } = false;

    /// <summary>Reuse an existing crop stem instead of creating a timestamped triplet.</summary>
    public bool OverwriteOutputs { get; set; } = false;

    /// <summary>Folder containing the model subfolders (wd14, grounding-dino, sam).</summary>
    /// Default lives under %APPDATA%\DatasetCropper\models so it works out of the box
    /// regardless of where the app is launched from.
    public string ModelsFolder { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DatasetCropper", "models");

    /// <summary>Which mask-generator model to use (SAM, CLIPSeg, or Rembg-human).</summary>
    public MaskModel MaskModel { get; set; } = MaskModel.Sam;

    /// <summary>Which caption model to use. Kept separate from <see cref="MaskModel"/>.</summary>
    public CaptionModel CaptionModel { get; set; } = CaptionModel.Wd14;

    /// <summary>
    /// Which detector runs concept/subject box detection: Grounding DINO (open
    /// vocabulary, default) or YOLO (COCO-80 closed set, fast and small).
    /// </summary>
    public DetectorModel DetectorModel { get; set; } = DetectorModel.GroundingDino;

    // --- Thresholds ---
    /// <summary>WD14 tag confidence threshold (keep tags &gt;= this).</summary>
    public double Wd14Threshold { get; set; } = 0.35;

    /// <summary>Grounding DINO box score threshold.</summary>
    public double DinoBoxThreshold { get; set; } = 0.30;

    /// <summary>Grounding DINO text token score threshold.</summary>
    public double DinoTextThreshold { get; set; } = 0.25;

    /// <summary>Minimum overlap (pixels, 0..1024) between adjacent 1024px tiles.
    /// Default 512 = 50% overlap. Higher → more tiles.</summary>
    public int MinTileOverlap { get; set; } = 512;

    /// <summary>Minimum source-image side length for subject crops. Defaults to 1024 pixels.</summary>
    public int MinimumSubjectResolution { get; set; } = 1024;

    // --- Processing stages ---
    // Checkboxes in the app's Process tab (above the log); all default ON.
    // Unchecking one makes the pipeline skip that stage.
    /// <summary>Run the Topaz upscale stage (also requires <see cref="UseTopaz"/>).</summary>
    public bool StageTopaz { get; set; } = true;

    /// <summary>Run Grounding DINO concept detection (off: crops go to 'unsorted').</summary>
    public bool StageDetect { get; set; } = true;

    /// <summary>Run subject (2nd-list) detection and subject crops.</summary>
    public bool StageSubjects { get; set; } = true;

    /// <summary>Run SAM masking (off: all masks are solid white).</summary>
    public bool StageMasks { get; set; } = true;

    /// <summary>Run WD14 captions (off: captions are empty).</summary>
    public bool StageCaptions { get; set; } = true;

    /// <summary>
    /// Fixed text written before the WD14 tags in every caption .txt (comma
    /// separated: &lt;prefix&gt;,&lt;tag1&gt;,&lt;tag2&gt;…). Empty by default (off).
    /// </summary>
    public string CaptionPrefix { get; set; } = "";
    /// <summary>Move near-duplicate images to the duplicates folder (default off).</summary>
    public bool StageDedupe { get; set; } = false;

    public bool StageMultiCrop { get; set; } = false;

    // --- Topaz ---
    public bool UseTopaz { get; set; } = false;

    /// <summary>Path to tpai.exe. Defaults to the standard install location.</summary>
    public string TopazPath { get; set; } = DatasetCropper.Services.TopazService.DefaultTpaiPath;

    /// <summary>
    /// Enhancement arguments passed to tpai.exe after -o/--overwrite (e.g. "--upscale",
    /// "--noise --sharpen", "--upscale scale=4"). Defaults to "--upscale".
    /// </summary>
    public string TopazArgs { get; set; } = "--upscale";

    // --- Runtime ---
    /// <summary>Prefer CUDA execution provider when available (requires the .Gpu package).</summary>
    public bool UseGpu { get; set; } = false;
}
