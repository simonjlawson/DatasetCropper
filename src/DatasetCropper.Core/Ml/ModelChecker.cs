using System.IO;

namespace DatasetCropper.Ml;

/// <summary>
/// Lightweight, session-free check of which model files are present in the
/// models folder. Lets the user see what's ready before pressing Start, without
/// paying the cost (or the GPU-driver side effects) of creating ONNX sessions.
/// </summary>
public static class ModelChecker
{
    private static readonly (string Name, string Subfolder, string[] Files)[] Models =
    {
        ("WD14", "wd14", new[] { "model.onnx", "selected_tags.csv" }),
        ("ViT-GPT2", "vit-gpt2", new[] { "encoder_model_quantized.onnx", "decoder_model_quantized.onnx", "vocab.json", "merges.txt" }),
        ("DINO", "grounding-dino", new[] { "model.onnx", "tokenizer.json" }),
        ("SAM", "sam", new[] { "image_encoder.onnx", "mask_decoder.onnx" }),
        ("CLIPSeg", "clipseg", new[] { "model.onnx", "merges.txt", "vocab.json" }),
        ("Rembg-human", "rembg", new[] { "u2net_human_seg.onnx" }),
        ("YOLO", "yolo", new[] { "yolov8n.onnx" }),
    };

    /// <summary>
    /// Returns one status line per model: "ready" when every expected file
    /// exists, otherwise "missing: &lt;files&gt;".
    /// </summary>
    public static List<(string Name, string Status)> Check(string modelsFolder)
    {
        var result = new List<(string, string)>();
        foreach (var (name, sub, files) in Models)
        {
            string dir = Path.Combine(modelsFolder, sub);
            var missing = files.Where(f => !File.Exists(Path.Combine(dir, f))).ToList();
            result.Add((name, missing.Count == 0 ? "ready" : "missing: " + string.Join(", ", missing)));
        }
        return result;
    }
}
