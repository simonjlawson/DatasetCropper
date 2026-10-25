using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using DatasetCropper.Models;

namespace DatasetCropper.Ml;

/// <summary>
/// Natural-language image captioning using Xenova/vit-gpt2-image-captioning.
/// The encoder runs once per crop and the GPT-2 decoder is greedily advanced
/// until end-of-text or the model's 20-token caption limit.
/// </summary>
public sealed class VitGpt2Captioner : ICaptioner
{
    private const int Side = 224;
    private const int MaxTokens = 20;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly Gpt2Tokenizer _tokenizer;
    private readonly string _encoderInput;
    private readonly string _encoderStatesOutput;
    private readonly string _decoderIds;
    private readonly string _decoderEncoderStates;
    private readonly string? _decoderEncoderMask;
    private readonly string? _decoderAttentionMask;
    private readonly string _logitsOutput;

    public string Name => "ViT-GPT2";

    private VitGpt2Captioner(
        InferenceSession encoder, InferenceSession decoder, Gpt2Tokenizer tokenizer,
        string encoderInput, string encoderStatesOutput, string decoderIds, string decoderEncoderStates,
        string? decoderEncoderMask, string? decoderAttentionMask, string logitsOutput)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _encoderInput = encoderInput;
        _encoderStatesOutput = encoderStatesOutput;
        _decoderIds = decoderIds;
        _decoderEncoderStates = decoderEncoderStates;
        _decoderEncoderMask = decoderEncoderMask;
        _decoderAttentionMask = decoderAttentionMask;
        _logitsOutput = logitsOutput;
    }

    public static VitGpt2Captioner? TryLoad(string modelDir, bool useGpu, out string? error)
    {
        error = null;
        string? encoderPath = FirstExisting(modelDir, "encoder_model_quantized.onnx", "encoder_model.onnx");
        string? decoderPath = FirstExisting(modelDir, "decoder_model_quantized.onnx", "decoder_model.onnx");
        string vocab = Path.Combine(modelDir, "vocab.json");
        string merges = Path.Combine(modelDir, "merges.txt");
        if (encoderPath is null || decoderPath is null)
        {
            error = "ViT-GPT2 encoder_model.onnx and decoder_model.onnx are required in " + modelDir;
            return null;
        }
        if (!File.Exists(vocab) || !File.Exists(merges))
        {
            error = "ViT-GPT2 vocab.json and merges.txt are required in " + modelDir;
            return null;
        }

        InferenceSession? encoder = null;
        InferenceSession? decoder = null;
        try
        {
            encoder = OnnxFactory.Create(Path.Combine(modelDir, encoderPath), useGpu);
            decoder = OnnxFactory.Create(Path.Combine(modelDir, decoderPath), useGpu);
            var tokenizer = new Gpt2Tokenizer(vocab, merges);

            string encoderInput = encoder.InputMetadata.First(p => p.Value.Dimensions.Length == 4).Key;
            string encoderStatesOutput = encoder.OutputMetadata.First(p => p.Value.Dimensions.Length == 3).Key;
            string decoderIds = decoder.InputMetadata.First(p => IsInteger(p.Value) && p.Value.Dimensions.Length == 2).Key;
            string decoderStates = decoder.InputMetadata.First(p => !IsInteger(p.Value) && p.Value.Dimensions.Length == 3).Key;
            string? encoderMask = decoder.InputMetadata.Keys.FirstOrDefault(n => n.Contains("encoder_attention", StringComparison.OrdinalIgnoreCase));
            string? attentionMask = decoder.InputMetadata.Keys.FirstOrDefault(n => n.Equals("attention_mask", StringComparison.OrdinalIgnoreCase));
            if (decoder.InputMetadata.Keys.Any(n => n.Contains("past", StringComparison.OrdinalIgnoreCase)))
                throw new InvalidDataException("decoder_model.onnx exposes past-key inputs; use the non-past decoder export.");

            string logits = decoder.OutputMetadata.FirstOrDefault(p => p.Key.Contains("logits", StringComparison.OrdinalIgnoreCase)).Key;
            if (string.IsNullOrEmpty(logits))
                logits = decoder.OutputMetadata.First(p => p.Value.Dimensions.Length >= 2).Key;

            return new VitGpt2Captioner(encoder, decoder, tokenizer, encoderInput, encoderStatesOutput, decoderIds,
                decoderStates, encoderMask, attentionMask, logits);
        }
        catch (Exception ex)
        {
            encoder?.Dispose();
            decoder?.Dispose();
            error = "could not load ViT-GPT2: " + ex.Message;
            return null;
        }
    }

    public string Caption(Image<Rgba32> image, double threshold)
    {
        using var resized = ImagePreprocess.Resize(image, Side, Side);
        byte[] rgb = ImagePreprocess.ToRgbBytes(resized);
        float[] pixels = ImagePreprocess.ToChwTensor(rgb, Side, Side, 1f / 255f, .5f, .5f, .5f, .5f, .5f, .5f);
        var input = new DenseTensor<float>(pixels, new[] { 1, 3, Side, Side });

        using var encoded = _encoder.Run(new[] { NamedOnnxValue.CreateFromTensor(_encoderInput, input) });
        var hidden = encoded.First(o => o.Name == _encoderStatesOutput).AsTensor<float>();
        float[] hiddenData = hidden.ToArray();
        int hiddenLength = hidden.Dimensions[^2];
        int hiddenSize = hidden.Dimensions[^1];
        var hiddenTensor = new DenseTensor<float>(hiddenData, new[] { 1, hiddenLength, hiddenSize });

        var ids = new List<long> { Gpt2Tokenizer.EndOfTextId };
        for (int step = 0; step < MaxTokens; step++)
        {
            using var outputs = RunDecoder(ids, hiddenTensor, hiddenLength);
            var logits = outputs.First(o => o.Name == _logitsOutput).AsTensor<float>();
            int vocabSize = logits.Dimensions[^1];
            int last = logits.Dimensions[^2] - 1;
            int best = 0;
            float bestScore = float.NegativeInfinity;
            for (int token = 0; token < vocabSize; token++)
            {
                float score = logits[0, last, token];
                if (score > bestScore) { bestScore = score; best = token; }
            }
            if (best == Gpt2Tokenizer.EndOfTextId) break;
            ids.Add(best);
        }
        return _tokenizer.Decode(ids.Skip(1));
    }

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunDecoder(
        List<long> ids, DenseTensor<float> hidden, int hiddenLength)
    {
        var feeds = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_decoderIds, new DenseTensor<long>(ids.ToArray(), new[] { 1, ids.Count })),
            NamedOnnxValue.CreateFromTensor(_decoderEncoderStates, hidden),
        };
        if (_decoderEncoderMask is not null)
            feeds.Add(NamedOnnxValue.CreateFromTensor(_decoderEncoderMask,
                new DenseTensor<long>(Enumerable.Repeat(1L, hiddenLength).ToArray(), new[] { 1, hiddenLength })));
        if (_decoderAttentionMask is not null)
            feeds.Add(NamedOnnxValue.CreateFromTensor(_decoderAttentionMask,
                new DenseTensor<long>(Enumerable.Repeat(1L, ids.Count).ToArray(), new[] { 1, ids.Count })));
        return _decoder.Run(feeds);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }

    private static bool IsInteger(NodeMetadata metadata)
        => metadata.ElementType == typeof(long) || metadata.ElementType == typeof(int);

    private static string? FirstExisting(string dir, params string[] names)
        => names.Select(n => Path.Combine(dir, n)).FirstOrDefault(File.Exists) is string path
            ? Path.GetFileName(path) : null;
}
