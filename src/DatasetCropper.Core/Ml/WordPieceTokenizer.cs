using System.Text;
using System.IO;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DatasetCropper.Ml;

/// <summary>
/// A minimal WordPiece (BERT-style) tokenizer that reads a Hugging Face
/// <c>tokenizer.json</c>. This is what the Grounding DINO export uses (a BERT
/// vocabulary with lowercase normalization and the <c>##</c> continuation prefix).
///
/// Produces the five text tensors the model expects:
///   input_ids, attention_mask, token_type_ids, position_ids, text_token_mask
/// </summary>
public sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly int _clsId, _sepId, _padId, _unkId;
    private readonly int _maxLength;

    public WordPieceTokenizer(string tokenizerJsonPath, int maxLength = 256)
    {
        _maxLength = maxLength;
        var json = File.ReadAllText(tokenizerJsonPath);
        var root = JsonNode.Parse(json)!;

        var model = root["model"]!;
        _vocab = new Dictionary<string, int>();
        var vocabNode = model["vocab"]!;
        foreach (var prop in vocabNode.AsObject())
            _vocab[prop.Key] = prop.Value!.GetValue<int>();

        _clsId = IdFor("[CLS]");
        _sepId = IdFor("[SEP]");
        _padId = IdFor("[PAD]");
        _unkId = IdFor("[UNK]");
    }

    private int IdFor(string token) => _vocab.TryGetValue(token, out int id) ? id : 0;

    /// <summary>
    /// Reverses a list of token ids back into a phrase (used to recover the
    /// matched text from a Grounding DINO query). Continuation tokens are
    /// concatenated; special tokens are dropped.
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds)
        {
            string? token = IdToToken(id);
            if (token is null) continue;
            if (token.StartsWith("##")) sb.Append(token[2..]);
            else if (sb.Length > 0) sb.Append(' ').Append(token);
            else sb.Append(token);
        }
        return sb.ToString().Trim();
    }

    private Dictionary<int, string>? _idToToken;
    private Dictionary<int, string> IdToTokenMap()
    {
        if (_idToToken is null)
        {
            _idToToken = new Dictionary<int, string>();
            foreach (var kv in _vocab)
                if (!_idToToken.ContainsKey(kv.Value)) _idToToken[kv.Value] = kv.Key;
        }
        return _idToToken;
    }

    private string? IdToToken(int id)
    {
        var map = IdToTokenMap();
        if (map.TryGetValue(id, out string? t)) return t;
        return null;
    }

    /// <summary>
    /// Tokenizes a Grounding-DINO style caption (lowercased, dot-separated
    /// concepts) into the model's text tensors.
    /// </summary>
    public (int[] inputIds, int[] attentionMask, int[] tokenTypeIds, int[] positionIds, int[] textTokenMask)
    Tokenize(string caption)
    {
        string text = caption.ToLowerInvariant().Trim();
        if (!text.EndsWith('.')) text += ".";

        // BertPreTokenizer: split on whitespace/punctuation boundaries.
        var pieces = PreTokenize(text);

        var ids = new List<int> { _clsId };
        var mask = new List<int> { 1 };
        var types = new List<int> { 0 };
        var tmask = new List<int> { 0 }; // [CLS] is not a word token

        foreach (var piece in pieces)
        {
            foreach (var sub in WordPieceEncode(piece))
            {
                int id = _vocab.TryGetValue(sub, out int v) ? v : _unkId;
                ids.Add(id);
                mask.Add(1);
                types.Add(0);
                // The '.' separator token is not a word token.
                tmask.Add(sub == "." ? 0 : 1);
            }
        }

        ids.Add(_sepId);
        mask.Add(1);
        types.Add(0);
        tmask.Add(0);

        // Truncate / pad to maxLength.
        int len = Math.Min(ids.Count, _maxLength);
        var inputIds = new int[_maxLength];
        var attention = new int[_maxLength];
        var tokenType = new int[_maxLength];
        var position = new int[_maxLength];
        var textMask = new int[_maxLength];
        for (int i = 0; i < _maxLength; i++)
        {
            if (i < len)
            {
                inputIds[i] = ids[i];
                attention[i] = mask[i];
                tokenType[i] = types[i];
                position[i] = i;
                textMask[i] = tmask[i];
            }
            else
            {
                inputIds[i] = _padId;
                attention[i] = 0;
                tokenType[i] = 0;
                position[i] = 0;
                textMask[i] = 0;
            }
        }
        return (inputIds, attention, tokenType, position, textMask);
    }

    /// <summary>
    /// BERT pre-tokenization: split on runs of whitespace and punctuation,
    /// keeping the punctuation as its own token.
    /// </summary>
    private static IEnumerable<string> PreTokenize(string text)
    {
        // Matches a word token (letters/digits incl. unicode) or a single punctuation char.
        var matches = Regex.Matches(text, @"[\p{L}\p{N}]+|[^\s\p{L}\p{N}]");
        foreach (Match m in matches)
            yield return m.Value;
    }

    /// <summary>
    /// Splits a single pre-tokenized piece into WordPiece subword tokens using
    /// the longest-match strategy (the standard BERT algorithm).
    /// </summary>
    private IEnumerable<string> WordPieceEncode(string piece)
    {
        if (piece.Length > 100) { yield return "[UNK]"; yield break; } // max_input_chars_per_word guard

        int i = 0;
        while (i < piece.Length)
        {
            // Longest-match: scan from the longest possible subword down to 1.
            int found = -1;
            for (int end = piece.Length; end > i; end--)
            {
                string sub = piece[i..end];
                if (i > 0) sub = "##" + sub;
                if (_vocab.ContainsKey(sub)) { found = end; break; }
            }

            if (found > i)
            {
                string sub = piece[i..found];
                yield return i > 0 ? "##" + sub : sub;
                i = found;
            }
            else
            {
                yield return "[UNK]";
                i++;
            }
        }
    }
}
