using System.Collections.Concurrent;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// Grounding DINO open-vocabulary detector, run in-process via ONNX.
///
/// Targets the onnx-community/grounding-dino-tiny-ONNX export (a single graph).
/// Inputs (fed by name-matching against the session, so minor export variants
/// still work):
///   pixel_values  (1,3,800,800)  RGB, /255, ImageNet-normalized
///   pixel_mask    (1,800,800)    ones
///   input_ids, attention_mask, token_type_ids, position_ids, text_token_mask
/// Outputs:
///   logits       (1,nq,256)
///   pred_boxes   (1,nq,4)  normalized cxcywh
///
/// Boxes are returned in the ORIGINAL image's pixel coordinates.
/// </summary>
public sealed class GroundingDinoDetector : IDetector
{
    public string Name => "Grounding DINO";

    private const int ImageSize = 800;

    private readonly InferenceSession? _session;
    private readonly WordPieceTokenizer? _tokenizer;
    private readonly KeyValuePair<string, NodeMetadata>[] _inputs;
    private readonly string _logitsName;
    private readonly string _boxesName;
    private readonly ConcurrentDictionary<string, (int[] inputIds, int[] attentionMask, int[] tokenTypeIds, int[] positionIds, int[] textTokenMask)> _tokenCache = new();

    private GroundingDinoDetector(InferenceSession? session, WordPieceTokenizer? tokenizer,
        KeyValuePair<string, NodeMetadata>[] inputs, string logitsName, string boxesName)
    {
        _session = session;
        _tokenizer = tokenizer;
        _inputs = inputs;
        _logitsName = logitsName;
        _boxesName = boxesName;
    }

    public static GroundingDinoDetector? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string modelPath = Path.Combine(modelDir, "model.onnx");
        string tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(modelPath)) { error = "model.onnx not found in " + modelDir; return null; }
        if (!File.Exists(tokenizerPath)) { error = "tokenizer.json not found in " + modelDir; return null; }

        var session = OnnxFactory.Create(modelPath, useGpu);
        var tokenizer = new WordPieceTokenizer(tokenizerPath);

        string logitsName = session.OutputNames.FirstOrDefault(n => n.Contains("logit", StringComparison.OrdinalIgnoreCase))
                            ?? session.OutputNames[0];
        string boxesName = session.OutputNames.FirstOrDefault(n => n.Contains("box", StringComparison.OrdinalIgnoreCase))
                           ?? session.OutputNames[1];

        return new GroundingDinoDetector(session, tokenizer, session.InputMetadata.Select(kv => kv).ToArray(), logitsName, boxesName);
    }

    public bool IsLoaded => _session is not null;

    /// <summary>
    /// Detects the given concepts in an image. Returns one <see cref="ConceptBox"/>
    /// per detection (in original-image pixel coords), each tagged with the matched
    /// concept.
    /// </summary>
    public List<ConceptBox> Detect(Image<Rgba32> image, IReadOnlyList<string> concepts,
        double boxThreshold, double textThreshold)
    {
        if (_session is null || _tokenizer is null || concepts.Count == 0) return new List<ConceptBox>();

        int origW = image.Width, origH = image.Height;

        // --- Image tensor (built once, shared across per-concept inferences) ---
        var resized = ImagePreprocess.Resize(image, ImageSize, ImageSize);
        byte[] rgb = ImagePreprocess.ToRgbBytes(resized, (255, 255, 255));
        float[] pixelValues = ImagePreprocess.ToChwTensor(
            rgb, ImageSize, ImageSize, rescale: 1f / 255f,
            meanR: 0.485f, meanG: 0.456f, meanB: 0.406f,
            stdR: 0.229f, stdG: 0.224f, stdB: 0.225f);

        // Grounding DINO routes every prompt token through one shared
        // cross-attention stream, so a strong concept suppresses weaker
        // co-prompted ones (e.g. a confident "man" drowns out a partially
        // occluded "face"). Detecting each concept in its own inference avoids
        // that cross-concept suppression at the cost of one extra run per concept.
        var distinct = concepts
            .Select(c => c.Trim().ToLowerInvariant())
            .Where(c => c.Length > 0)
            .Distinct()
            .ToList();
        var all = new List<ConceptBox>();
        foreach (var concept in distinct)
            all.AddRange(DetectOne(pixelValues, origW, origH, concept, boxThreshold, textThreshold));
        return all;
    }

    /// <summary>
    /// Runs one inference for a single <paramref name="concept"/> and returns its
    /// boxes (NMSed). The image tensor is supplied precomputed so multi-concept
    /// detection can reuse it across inferences.
    /// </summary>
    private List<ConceptBox> DetectOne(float[] pixelValues, int origW, int origH,
        string concept, double boxThreshold, double textThreshold)
    {
        var result = new List<ConceptBox>();

        // --- Text tensors for this single concept ---
        var toks = _tokenCache!.GetOrAdd(concept, _ => _tokenizer!.Tokenize(concept));

        // --- Build inputs, honouring each input's declared element type. ---
        // Different ONNX exports declare their text-side inputs differently:
        //   onnx-community/grounding-dino-tiny: Int64 pixel_mask, Int64 text tensors
        //   huggingface/grounding-dino:        Float/Bool pixel_mask, Float text tensors
        // Feeding the wrong element type raises InvalidArgument like
        //   "Tensor element data type discovered: Float metadata expected: Int64"
        //   "Tensor element data type discovered: Int32 metadata expected: Int64"
        // We therefore inspect each input's declared System.Type via
        // NodeMetadata.ElementType and feed a tensor of matching element type.
        var feeds = new List<NamedOnnxValue>();

        void AddLong(string name, int[] ints, int[] shape)
        {
            // Widen int -> long: token ids and 0/1 flags all fit trivially.
            var longs = new long[ints.Length];
            for (int i = 0; i < ints.Length; i++) longs[i] = ints[i];
            feeds.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(longs, shape)));
        }
        void AddFloatArr(string name, float[] f, int[] shape)
        {
            feeds.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(f, shape)));
        }
        void AddBool(string name, int[] bits, int[] shape)
        {
            var booleans = new bool[bits.Length];
            for (int i = 0; i < bits.Length; i++) booleans[i] = bits[i] != 0;
            feeds.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<bool>(booleans, shape)));
        }
        // Dispatch a single input by its declared element type. The value arrays
        // passed in are int-based (0/1 flags or ids) and widened/narrowed to
        // match what the graph expects. DenseTensor only supports the
        // element types the C# runtime exposes (int, long, float, bool), so
        // int32 / int64 / bool / float are the only branches we need.
        void Add(string name, int[] ints, int[] shape, Type declared)
        {
            if (declared == typeof(float)) AddFloatArr(name, ints.Select(v => (float)v).ToArray(), shape);
            else if (declared == typeof(bool)) AddBool(name, ints, shape);
            else if (declared == typeof(int))
            {
                var t = new DenseTensor<int>(ints, shape);
                feeds.Add(NamedOnnxValue.CreateFromTensor(name, t));
            }
            else AddLong(name, ints, shape); // long / int64 / default
        }

        foreach (var kv in _inputs)
        {
            Type declared = kv.Value.ElementType ?? typeof(float);
            string name = kv.Key;

            if (name.Contains("pixel_values", StringComparison.OrdinalIgnoreCase))
            {
                // Always float — pixel input values.
                AddFloatArr(name, pixelValues, new[] { 1, 3, ImageSize, ImageSize });
            }
            else if (name.Contains("pixel_mask", StringComparison.OrdinalIgnoreCase))
            {
                // A validity mask (==1 keeps the pixel); some exports declare it
                // float, others int64 (and a few bool). Feed a matching tensor.
                Add(name, Enumerable.Repeat(1, ImageSize * ImageSize).ToArray(), new[] { 1, ImageSize, ImageSize }, declared);
            }
            else if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase))
            {
                // Token ids — always an integer type, but dispatch by declared
                // type so Int32 vs Int64 variants both work.
                Add(name, toks.inputIds, new[] { 1, toks.inputIds.Length }, declared);
            }
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase))
            {
                Add(name, toks.attentionMask, new[] { 1, toks.attentionMask.Length }, declared);
            }
            else if (name.Contains("token_type_ids", StringComparison.OrdinalIgnoreCase))
            {
                Add(name, toks.tokenTypeIds, new[] { 1, toks.tokenTypeIds.Length }, declared);
            }
            else if (name.Contains("position_ids", StringComparison.OrdinalIgnoreCase))
            {
                Add(name, toks.positionIds, new[] { 1, toks.positionIds.Length }, declared);
            }
            else if (name.Contains("text_token_mask", StringComparison.OrdinalIgnoreCase))
            {
                Add(name, toks.textTokenMask, new[] { 1, toks.textTokenMask.Length }, declared);
            }
            // Unknown input: skip (session may have optional inputs).
        }

        using var outputs = _session.Run(feeds.ToArray());

        // --- Read outputs ---
        var logits = outputs.FirstOrDefault(o => o.Name == _logitsName)?.AsTensor<float>()
                    ?? throw new InvalidOperationException("logits output missing");
        var boxes = outputs.FirstOrDefault(o => o.Name == _boxesName)?.AsTensor<float>()
                   ?? throw new InvalidOperationException("pred_boxes output missing");

        int[] dims = logits.Dimensions.ToArray();
        int nq = dims[1];
        int textLen = dims[2]; // number of token positions (e.g. 256)

        // Sigmoid the logits, compute per-query max score over token positions.
        var scores = new float[nq];
        for (int q = 0; q < nq; q++)
        {
            float max = float.MinValue;
            for (int t = 0; t < textLen; t++)
            {
                float s = Sigmoid(logits[0, q, t]);
                if (s > max) max = s;
            }
            scores[q] = max;
        }

        // Single-concept inference: the only candidate is the one we prompted.
        var lowerConcepts = new List<string> { concept };

        for (int q = 0; q < nq; q++)
        {
            if (scores[q] <= boxThreshold) continue;

            // Recover the phrase: tokens whose logit exceeds text_threshold.
            var tokenIds = new List<int>();
            for (int t = 0; t < textLen; t++)
                if (Sigmoid(logits[0, q, t]) > textThreshold)
                    tokenIds.Add(toks.inputIds[t]);
            string phrase = _tokenizer!.Decode(tokenIds.ToArray()).Replace('.', ' ').Replace("  ", " ").Trim();

            string matched = MatchConcept(phrase, lowerConcepts);
            if (matched.Length == 0) continue;

            // cxcywh (normalized) -> xyxy in original pixels.
            float cx = boxes[0, q, 0], cy = boxes[0, q, 1], bw = boxes[0, q, 2], bh = boxes[0, q, 3];
            float x1 = (cx - bw / 2f) * origW;
            float y1 = (cy - bh / 2f) * origH;
            float x2 = (cx + bw / 2f) * origW;
            float y2 = (cy + bh / 2f) * origH;
            x1 = Math.Clamp(x1, 0, origW); y1 = Math.Clamp(y1, 0, origH);
            x2 = Math.Clamp(x2, 0, origW); y2 = Math.Clamp(y2, 0, origH);
            if (x2 - x1 < 2 || y2 - y1 < 2) continue;

            result.Add(new ConceptBox
            {
                Concept = matched,
                X = x1, Y = y1, W = x2 - x1, H = y2 - y1,
                Score = scores[q]
            });
        }

        // NMS-ish: drop boxes that are near-duplicates of a higher-scoring box.
        result.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<ConceptBox>();
        foreach (var b in result)
        {
            if (!kept.Any(k => k.Concept == b.Concept && IoU(k, b) > 0.5f))
                kept.Add(b);
        }
        return kept;
    }

    private static string MatchConcept(string phrase, List<string> lowerConcepts)
    {
        string p = phrase.ToLowerInvariant();
        string best = "";
        foreach (var c in lowerConcepts)
            if (p.Contains(c) && c.Length > best.Length)
                best = c;
        return best.Length > 0 ? best : SanitizeConcept(phrase);
    }

    private static string SanitizeConcept(string phrase)
    {
        var words = phrase.Split(' ', StringSplitOptions.RemoveEmptyEntries).Take(4).ToArray();
        return string.Join(' ', words);
    }

    private static float IoU(ConceptBox a, ConceptBox b)
    {
        float ix1 = Math.Max(a.X, b.X), iy1 = Math.Max(a.Y, b.Y);
        float ix2 = Math.Min(a.X2, b.X2), iy2 = Math.Min(a.Y2, b.Y2);
        float inter = Math.Max(0, ix2 - ix1) * Math.Max(0, iy2 - iy1);
        float uni = a.W * a.H + b.W * b.H - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public void Dispose() => _session?.Dispose();
}
