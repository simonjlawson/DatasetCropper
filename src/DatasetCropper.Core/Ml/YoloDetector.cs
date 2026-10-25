using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// YOLOv8 / YOLO11 (COCO-80 class) detector, run in-process via ONNX.
///
/// Fast, small alternative to Grounding DINO for subject/concept detection: one
/// inference pass (no per-concept token runs) and a small model. The trade-off is
/// that it is <b>closed-vocabulary</b> &mdash; the standard 80 COCO classes &mdash;
/// so requested concepts are mapped to their nearest COCO class, and a concept with
/// no COCO analogue (e.g. "face" is approximated as "person", but "sailboat" has
/// none) simply won't be found. Use Grounding DINO for fully open concepts.
///
/// Pre-processing (ultralytics reference): letterbox-resize to a 640&times;640
/// gray-padded square, RGB /255 (no ImageNet normalization), CHW, NCHW. Boxes come
/// back in that 640-space; we unletterbox here so results are in the input image's
/// own pixels. NMS runs in-process (the reference ONNX exports have no NMS baked
/// in), so the layout of the raw candidate tensor can vary by export and is
/// matched defensively at load/infer time.
/// </summary>
public sealed class YoloDetector : IDetector
{
    private const int ImageSize = 640;
    private const byte Pad = 114;

    /// <summary>Standard COCO-80 class names, in index order.</summary>
    private static readonly string[] CocoClasses =
    {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
        "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
        "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };

    /// <summary>
    /// Loose synonyms so common user phrasings land on a COCO class (mostly the
    /// person/vehicle/animal words that subject detection cares about).
    /// </summary>
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["people"] = "person",  ["man"] = "person",  ["woman"] = "person", ["girl"] = "person",
        ["boy"] = "person",  ["kid"] = "person",  ["child"] = "person", ["face"] = "person",
        ["head"] = "person", ["human"] = "person", ["figure"] = "person",
        ["auto"] = "car",  ["automobile"] = "car", ["vehicle"] = "car", ["sedan"] = "car",
        ["lorry"] = "truck",
        ["motorbike"] = "motorcycle", ["scooter"] = "motorcycle",
        ["bike"] = "bicycle", ["cycle"] = "bicycle",
        ["ship"] = "boat",  ["yacht"] = "boat",
        ["plane"] = "airplane", ["aircraft"] = "airplane",
        ["kitten"] = "cat", ["kitty"] = "cat", ["feline"] = "cat",
        ["puppy"] = "dog", ["pup"] = "dog", ["canine"] = "dog", ["hound"] = "dog",
        ["laptop"] = "laptop", ["notebook"] = "laptop",
        ["phone"] = "cell phone", ["mobile"] = "cell phone", ["smartphone"] = "cell phone",
    };

    public string Name => "YOLO";
    public bool IsLoaded => _session is not null;

    private readonly InferenceSession? _session;
    private readonly string _imageIn;

    private YoloDetector(InferenceSession session, string imageIn)
    {
        _session = session;
        _imageIn = imageIn;
    }

    /// <summary>
    /// Attempts to load a YOLO detection ONNX (yolov8*.onnx / model.onnx / best.onnx
    /// / the largest .onnx in the folder). Returns null with a reason in
    /// <paramref name="error"/> otherwise.
    /// </summary>
    public static YoloDetector? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string? dir = SafeExists(modelDir) ? modelDir : null;
        if (dir is null) { error = "models folder not found: " + modelDir; return null; }

        string? onnxFile = FirstExisting(modelDir,
                "yolov8n.onnx", "yolov8s.onnx", "yolov8m.onnx", "yolov11n.onnx", "yolov11s.onnx",
                "model.onnx", "best.onnx", "yolov8.onnx")
            ?? SafeList(modelDir, "*.onnx").OrderByDescending(SafeLength).Cast<string?>().FirstOrDefault();
        if (onnxFile is null) { error = "no YOLO ONNX found in " + modelDir; return null; }

        InferenceSession session;
        try
        {
            session = OnnxFactory.Create(Path.Combine(modelDir, onnxFile), useGpu);
        }
        catch (Exception ex)
        {
            error = "could not create the YOLO session for " + onnxFile + ": " + ex.Message;
            return null;
        }

        string imageIn = session.InputMetadata
            .FirstOrDefault(im => im.Value.Dimensions.Length == 4).Key
            ?? session.InputMetadata.First().Key;

        return new YoloDetector(session, imageIn);
    }

    /// <inheritdoc/>
    public List<ConceptBox> Detect(Image<Rgba32> image, IReadOnlyList<string> concepts,
        double boxThreshold, double textThreshold)
    {
        if (_session is null || concepts.Count == 0) return new List<ConceptBox>();
        int srcW = image.Width, srcH = image.Height;

        // Which COCO classes each requested concept maps to; skip concepts with no
        // closed-vocabulary analogue (they cannot be detected by this model).
        var conceptToClasses = new List<(string concept, List<int> classes)>();
        foreach (var c in concepts)
        {
            var cl = MapConceptToClasses(c);
            if (cl.Count > 0) conceptToClasses.Add((c, cl));
        }
        if (conceptToClasses.Count == 0) return new List<ConceptBox>();

        // --- Letterbox pre-process (RGB /255, no ImageNet normalization) ---
        float scale = MathF.Min(ImageSize / (float)srcW, ImageSize / (float)srcH);
        int newW = Math.Max(1, (int)MathF.Round(srcW * scale));
        int newH = Math.Max(1, (int)MathF.Round(srcH * scale));
        int padX = (ImageSize - newW) / 2;
        int padY = (ImageSize - newH) / 2;

        using var resized = ImagePreprocess.Resize(image, newW, newH);
        using var canvas = ImagePreprocess.CompositeOnCanvas(resized, ImageSize, ImageSize, padX, padY, (Pad, Pad, Pad));
        byte[] rgb = ImagePreprocess.ToRgbBytes(canvas, (114, 114, 114));
        float[] chw = ImagePreprocess.ToChwTensor(rgb, ImageSize, ImageSize, rescale: 1f / 255f);

        using var outputs = _session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(_imageIn,
                new DenseTensor<float>(chw, new[] { 1, 3, ImageSize, ImageSize }))
        });

        // --- Parse raw detections (cx, cy, w, h in 640-space; score; class) ---
        var raw = ParseDetections(outputs).ToList();

        // --- Keep only candidates the user asked for, above threshold ---
        var picked = new List<ConceptBox>();
        foreach (var (cx, cy, w, h, score, cls) in raw)
        {
            if (score < boxThreshold) continue;
            var owner = conceptToClasses.FirstOrDefault(kv2 => kv2.classes.Contains(cls));
            if (owner.concept is null) continue;

            // Unletterbox: 640-space -> source-image pixels.
            float x = (cx - (w / 2f) - padX) / scale;
            float y = (cy - (h / 2f) - padY) / scale;
            float ww = w / scale, hh = h / scale;
            float x1 = Math.Clamp(x, 0, srcW), y1 = Math.Clamp(y, 0, srcH);
            float x2 = Math.Clamp(x + ww, 0, srcW), y2 = Math.Clamp(y + hh, 0, srcH);
            if (x2 - x1 < 2 || y2 - y1 < 2) continue;

            picked.Add(new ConceptBox
            {
                Concept = owner.concept,
                X = x1, Y = y1, W = x2 - x1, H = y2 - y1,
                Score = (float)score
            });
        }

        // NMS per concept (drop near-duplicates of a higher-scoring box).
        picked.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<ConceptBox>();
        foreach (var b in picked)
            if (!kept.Any(k => k.Concept == b.Concept && IoU(k, b) > 0.5f))
                kept.Add(b);
        return kept;
    }

    // ---- parsing / matching helpers ----

    private static IEnumerable<(float cx, float cy, float w, float h, float score, int cls)> ParseDetections(
        System.Collections.Generic.IEnumerable<NamedOnnxValue> outputs)
    {
        var t3 = outputs.Select(o => o.AsTensor<float>()).Where(t => t.Dimensions.Length == 3).ToList();

        // ---- Single fused tensor [1, C, N] (C = 4 + numClasses), the common YOLO ONNX layout ----
        if (t3.Count == 1)
        {
            Tensor<float> f = t3[0];
            int[] d = f.Dimensions.ToArray();
            bool cFirst;
            int C, N;
            if (d[1] <= d[2]) { cFirst = true;  C = d[1]; N = d[2]; }
            else              { cFirst = false; C = d[2]; N = d[1]; }
            int nc = C - 4;
            if (nc <= 0) throw new InvalidOperationException($"YOLO detector: fused output has too few channels ({C}).");
            for (int i = 0; i < N; i++)
            {
                float cx, cy, w, h, best = 0f; int bc = 0;
                if (cFirst)
                {
                    cx = f[0, 0, i]; cy = f[0, 1, i]; w = f[0, 2, i]; h = f[0, 3, i];
                    for (int c = 0; c < nc; c++) { float s = f[0, 4 + c, i]; if (s > best) { best = s; bc = c; } }
                }
                else
                {
                    cx = f[0, i, 0]; cy = f[0, i, 1]; w = f[0, i, 2]; h = f[0, i, 3];
                    for (int c = 0; c < nc; c++) { float s = f[0, i, 4 + c]; if (s > best) { best = s; bc = c; } }
                }
                yield return (cx, cy, w, h, best, bc);
            }
            yield break;
        }

        // ---- Two tensors: boxes [1, N, 4] | [1, 4, N]  and  classes [1, nc, N] | [1, N, nc] ----
        if (t3.Count == 2)
        {
            Tensor<float> boxes, classes;
            if (Is4Side(t3[0])) { boxes = t3[0]; classes = t3[1]; }
            else if (Is4Side(t3[1])) { boxes = t3[1]; classes = t3[0]; }
            else throw new InvalidOperationException("YOLO detector: could not identify the boxes/classes outputs.");

            int[] bd = boxes.Dimensions.ToArray();
            bool bxLast = bd[1] == 4;                       // [1, 4, N]
            int n = bxLast ? bd[2] : bd[1];

            int[] cd = classes.Dimensions.ToArray();
            int nc = (cd[1] == n) ? cd[2] : cd[1];          // the side that is NOT N
            bool clsCfirst = (cd[1] != n);                  // [1, nc, N]

            for (int i = 0; i < n; i++)
            {
                float cx, cy, w, h;
                if (bxLast) { cx = boxes[0, 0, i]; cy = boxes[0, 1, i]; w = boxes[0, 2, i]; h = boxes[0, 3, i]; }
                else        { cx = boxes[0, i, 0]; cy = boxes[0, i, 1]; w = boxes[0, i, 2]; h = boxes[0, i, 3]; }
                float best = 0f; int bc = 0;
                for (int c = 0; c < nc; c++)
                {
                    float s = clsCfirst ? classes[0, c, i] : classes[0, i, c];
                    if (s > best) { best = s; bc = c; }
                }
                yield return (cx, cy, w, h, best, bc);
            }
            yield break;
        }

        throw new InvalidOperationException(
            "YOLO detector: unexpected ONNX output layout (" + t3.Count +
            (t3.Count > 0 ? "; first dims " + string.Join(",", t3[0].Dimensions.ToArray()) : "") + ")");
    }

    private static bool Is4Side(Tensor<float> t)
        => t.Dimensions.Length == 3 && (t.Dimensions[1] == 4 || t.Dimensions[2] == 4);

    /// <summary>Maps a requested concept to the COCO class indices it can satisfy.</summary>
    private List<int> MapConceptToClasses(string concept)
    {
        var set = new List<int>();
        string key = (concept ?? "").Trim();
        if (key.Length == 0) return set;

        string? canon = Aliases.TryGetValue(key, out var a) ? a : null;
        // Handle a naive plural ("people" / "persons") on the alias keys too.
        if (canon is null && key.EndsWith('s') && Aliases.TryGetValue(key[..^1], out var a2)) canon = a2;

        string target = (canon ?? key).ToLowerInvariant();
        for (int i = 0; i < CocoClasses.Length; i++)
        {
            string coco = CocoClasses[i].ToLowerInvariant();
            if (coco == target) { set.Add(i); break; }
        }
        // Fallback: containment in either direction (e.g. "police car" contains "car").
        if (set.Count == 0)
        {
            for (int i = 0; i < CocoClasses.Length; i++)
            {
                string coco = CocoClasses[i].ToLowerInvariant();
                if (coco.Length >= 3 && (target.Contains(coco) || coco.Contains(target)))
                    set.Add(i);
            }
        }
        return set;
    }

    private static float IoU(ConceptBox a, ConceptBox b)
    {
        float ix1 = Math.Max(a.X, b.X), iy1 = Math.Max(a.Y, b.Y);
        float ix2 = Math.Min(a.X2, b.X2), iy2 = Math.Min(a.Y2, b.Y2);
        float inter = Math.Max(0, ix2 - ix1) * Math.Max(0, iy2 - iy1);
        float uni = a.W * a.H + b.W * b.H - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    private static bool SafeExists(string dir)
    {
        try { return Directory.Exists(dir); } catch { return false; }
    }

    private static string? FirstExisting(string dir, params string[] names)
    {
        foreach (var n in names)
            if (n is not null && File.Exists(Path.Combine(dir, n))) return n;
        return null;
    }

    private static IEnumerable<string> SafeList(string dir, string pattern)
    {
        try { return Directory.EnumerateFiles(dir, pattern).ToArray(); }
        catch { return Array.Empty<string>(); }
    }

    private static long SafeLength(string file)
    {
        try { return new FileInfo(file).Length; } catch { return 0; }
    }

    public void Dispose() => _session?.Dispose();
}
