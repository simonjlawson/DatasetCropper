using System.IO;
using DatasetCropper.Ml;
using Xunit;

namespace DatasetCropper.Tests;

public class ModelCheckerTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "DatasetCropper_checker_" + Guid.NewGuid().ToString("N"));

    public ModelCheckerTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { /* best effort */ }
    }

    private void MakeModelBundle(string name, bool full)
    {
        string dir = Path.Combine(_root, name);
        Directory.CreateDirectory(dir);
        // Minimum files the checker looks for per bundle.
        if (name == "wd14")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "model.onnx"), "");
            if (full) File.WriteAllText(Path.Combine(dir, "selected_tags.csv"), "");
        }
        else if (name == "grounding-dino")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "model.onnx"), "");
            if (full) File.WriteAllText(Path.Combine(dir, "tokenizer.json"), "{}");
        }
        else if (name == "sam")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "image_encoder.onnx"), "");
            if (full) File.WriteAllText(Path.Combine(dir, "mask_decoder.onnx"), "");
        }
        else if (name == "clipseg")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "model.onnx"), "");
            if (full) File.WriteAllText(Path.Combine(dir, "merges.txt"), "");
            if (full) File.WriteAllText(Path.Combine(dir, "vocab.json"), "{}");
        }
        else if (name == "rembg")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "u2net_human_seg.onnx"), "");
        }
        else if (name == "yolo")
        {
            if (full) File.WriteAllText(Path.Combine(dir, "yolov8n.onnx"), "");
        }
    }

    [Fact]
    public void Check_EmptyFolder_AllModelsMissing()
    {
        // 7 bundled models are checked, including the ViT-GPT2 captioner.
        var results = ModelChecker.Check(_root);

        Assert.Equal(7, results.Count);
    }

    [Fact]
    public void Check_EmptyFolder_Wd14MissingBothFiles()
    {
        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Contains("model.onnx", wd14.Status);
    }

    [Fact]
    public void Check_EmptyFolder_Wd14MissingCsv()
    {
        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Contains("selected_tags.csv", wd14.Status);
    }

    [Fact]
    public void Check_EmptyFolder_DinoMissingBothFiles()
    {
        var results = ModelChecker.Check(_root);
        var dino = results.First(r => r.Name == "DINO");

        Assert.Contains("model.onnx", dino.Status);
    }

    [Fact]
    public void Check_EmptyFolder_SamMissingBothFiles()
    {
        var results = ModelChecker.Check(_root);
        var sam = results.First(r => r.Name == "SAM");

        Assert.Contains("image_encoder.onnx", sam.Status);
    }

    [Fact]
    public void Check_AllBundlesComplete_AllReady()
    {
        MakeModelBundle("wd14", full: true);
        MakeModelBundle("grounding-dino", full: true);
        MakeModelBundle("sam", full: true);
        MakeModelBundle("clipseg", full: true);
        MakeModelBundle("rembg", full: true);
        MakeModelBundle("yolo", full: true);

        var results = ModelChecker.Check(_root);

        Assert.Equal(7, results.Count);
    }

    [Fact]
    public void Check_AllBundlesComplete_EachReportedReady()
    {
        MakeModelBundle("wd14", full: true);
        MakeModelBundle("grounding-dino", full: true);
        MakeModelBundle("sam", full: true);

        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Equal("ready", wd14.Status);
    }

    [Fact]
    public void Check_Wd14OnlyModelMissingCsv()
    {
        // Bundle dir exists, model.onnx present, selected_tags.csv absent.
        MakeModelBundle("wd14", full: false);
        File.WriteAllText(Path.Combine(_root, "wd14", "model.onnx"), "");

        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Contains("selected_tags.csv", wd14.Status);
    }

    [Fact]
    public void Check_Wd14OnlyCsvMissingModel()
    {
        // Bundle dir exists, selected_tags.csv present, model.onnx absent.
        MakeModelBundle("wd14", full: false);
        File.WriteAllText(Path.Combine(_root, "wd14", "selected_tags.csv"), "");

        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Contains("model.onnx", wd14.Status);
    }

    [Fact]
    public void Check_Wd14BothPresent_ReportedReady()
    {
        MakeModelBundle("wd14", full: true);

        var results = ModelChecker.Check(_root);
        var wd14 = results.First(r => r.Name == "WD14");

        Assert.Equal("ready", wd14.Status);
    }

    [Fact]
    public void Check_SamOnlyImageEncoderMissingMaskDecoder()
    {
        MakeModelBundle("sam", full: false);
        File.WriteAllText(Path.Combine(_root, "sam", "image_encoder.onnx"), "");

        var results = ModelChecker.Check(_root);
        var sam = results.First(r => r.Name == "SAM");

        Assert.Contains("mask_decoder.onnx", sam.Status);
    }
}
