using System.Globalization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;

namespace DatasetCropper.Ml;

/// <summary>
/// WD14 (waifu-diffusion v1.4) booru DatasetCropper, run in-process via ONNX.
///
/// Mirrors the reference preprocessing exactly:
///   alpha over white -> RGB -> BGR -> square pad (white) -> smart_resize ->
///   float32, NO /255, NO normalization.
///
/// The channel layout is read from the model, not assumed: some exports are
/// CHW <c>(1,3,H,W)</c> and others NHWC <c>(1,H,W,3)</c>. The reference feeds
/// BGR (cv2 order), so the channel order is BGR in either layout.
/// Output is raw softmax confidences; the first 4 rows of the tag CSV are the
/// ratings (general/sensitive/questionable/explicit), the rest are tags.
/// </summary>
public sealed class Wd14DatasetCropper : ICaptioner
{
    public string Name => "WD14";

    private readonly InferenceSession? _session;
    private readonly string[] _tagNames;
    private readonly string _inputName;
    private readonly int _inputH;
    private readonly int _inputW;
    private readonly bool _nhwc;

    private Wd14DatasetCropper(InferenceSession? session, string[] tagNames, string inputName, int inputH, int inputW, bool nhwc)
    {
        _session = session;
        _tagNames = tagNames;
        _inputName = inputName;
        _inputH = inputH;
        _inputW = inputW;
        _nhwc = nhwc;
    }

    /// <summary>
    /// Loads the model + tag CSV. Returns null (and sets <paramref name="error"/>)
    /// when the model or CSV is missing so the caller can skip the stage.
    /// </summary>
    public static Wd14DatasetCropper? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string modelPath = Path.Combine(modelDir, "model.onnx");
        string csvPath = Path.Combine(modelDir, "selected_tags.csv");
        if (!File.Exists(modelPath)) { error = "model.onnx not found in " + modelDir; return null; }
        if (!File.Exists(csvPath)) { error = "selected_tags.csv not found in " + modelDir; return null; }

        var tags = LoadTagNames(csvPath);
        var session = OnnxFactory.Create(modelPath, useGpu);
        string inputName = session.InputNames[0];
        int[] dims = session.InputMetadata[inputName].Dimensions;
        // Detect channel layout: NHWC (1,H,W,3) when the last dim is the 3
        // channels and the first spatial dim is not; otherwise CHW (1,3,H,W).
        bool nhwc = dims.Length == 4 && dims[3] == 3 && dims[1] != 3;
        int h = nhwc ? dims[1] : (dims.Length >= 3 ? dims[2] : 0);
        int w = nhwc ? dims[2] : (dims.Length >= 4 ? dims[3] : 0);
        return new Wd14DatasetCropper(session, tags, inputName, h, w, nhwc);
    }

    private static string[] LoadTagNames(string csvPath)
    {
        var names = new List<string>();
        using var reader = new StreamReader(csvPath);
        string? header = reader.ReadLine();
        int nameCol = 1; // default: tag_id,name,...
        if (header is not null)
        {
            var cols = header.Split(',');
            for (int i = 0; i < cols.Length; i++)
                if (cols[i].Trim().Equals("name", StringComparison.OrdinalIgnoreCase)) { nameCol = i; break; }
        }
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var cols = line.Split(',');
            if (nameCol < cols.Length) names.Add(cols[nameCol].Trim());
        }
        return names.ToArray();
    }

    public bool IsLoaded => _session is not null;

    /// <summary>
    /// Tags an image, returning the comma-joined caption of tags with confidence
    /// &gt;= threshold, sorted by confidence descending.
    /// </summary>
    public string Tag(Image<Rgba32> image, double threshold)
    {
        if (_session is null) return "";
        float[] input = Preprocess(image);
        var inputTensor = new DenseTensor<float>(input, _nhwc ? [1, _inputH, _inputW, 3] : [1, 3, _inputH, _inputW]);
        var outputs = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)]);
        var outputTensor = outputs.First().AsTensor<float>(); // (1, N)
        int n = outputTensor.Dimensions[1];
        var conf = new float[n];
        for (int i = 0; i < n; i++) conf[i] = outputTensor[0, i];

        var tagged = new List<(string Name, float Conf)>();
        for (int i = 4; i < _tagNames.Length && i < conf.Length; i++)
        {
            float c = conf[i];
            if (c >= threshold) tagged.Add((_tagNames[i], c));
        }
        tagged.Sort((a, b) => b.Conf.CompareTo(a.Conf));
        return string.Join(", ", tagged.Select(t => t.Name));
    }

    string ICaptioner.Caption(Image<Rgba32> image, double threshold) => Tag(image, threshold);

    private float[] Preprocess(Image<Rgba32> image)
    {
        // 1) alpha over white -> RGB bytes
        byte[] rgb = ImagePreprocess.ToRgbBytes(image, (255, 255, 255));
        int w = image.Width, h = image.Height;

        // 2) make_square: pad to max(w,h,input) with white
        int desired = Math.Max(Math.Max(w, h), Math.Max(_inputH, _inputW));
        int top = (desired - h) / 2, left = (desired - w) / 2;
        var square = ImagePreprocess.CompositeOnCanvas(
            ImagePreprocess.NewFromBytes(rgb, w, h), desired, desired, left, top, (255, 255, 255));

        // 3) smart_resize to the model's input size
        var resized = ImagePreprocess.Resize(square, _inputH, _inputW);

        // 4) RGB -> BGR, float32, no normalization. Channel order is BGR in both
        //    layouts; only the memory arrangement differs (NHWC per-pixel vs CHW planes).
        byte[] px = ImagePreprocess.ToRgbBytes(resized, (255, 255, 255));
        int plane = _inputH * _inputW;
        var tensor = new float[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            float b = px[i * 3 + 2];
            float g = px[i * 3 + 1];
            float r = px[i * 3 + 0];
            if (_nhwc)
            {
                tensor[i * 3 + 0] = b;
                tensor[i * 3 + 1] = g;
                tensor[i * 3 + 2] = r;
            }
            else
            {
                tensor[i] = b;
                tensor[plane + i] = g;
                tensor[2 * plane + i] = r;
            }
        }
        return tensor;
    }

    public void Dispose() => _session?.Dispose();
}
