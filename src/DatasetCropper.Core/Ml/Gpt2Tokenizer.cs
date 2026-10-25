using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DatasetCropper.Ml;

/// <summary>
/// Minimal GPT-2 byte-level BPE tokenizer for the ViT-GPT2 caption model.
/// It consumes the standard Hugging Face vocab.json and merges.txt files.
/// </summary>
public sealed class Gpt2Tokenizer
{
    public const int EndOfTextId = 50256;

    private static readonly Regex Tokens = new(
        "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string A, string B), int> _ranks;
    private readonly Dictionary<char, byte> _charToByte;
    private readonly char[] _byteToChar;

    public Gpt2Tokenizer(string vocabPath, string mergesPath)
    {
        _vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath))
            ?? throw new InvalidDataException("GPT-2 vocab.json is empty or invalid.");
        _idToToken = _vocab.ToDictionary(p => p.Value, p => p.Key);

        _ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var line in File.ReadLines(mergesPath))
        {
            var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2 || line.StartsWith('#')) continue;
            _ranks.TryAdd((parts[0], parts[1]), rank++);
        }

        (_byteToChar, _charToByte) = BuildByteMaps();
    }

    public List<long> Encode(string text)
    {
        var ids = new List<long>();
        foreach (Match match in Tokens.Matches(text))
        {
            string encoded = string.Concat(Encoding.UTF8.GetBytes(match.Value).Select(b => _byteToChar[b]));
            var pieces = encoded.Select(c => c.ToString()).ToList();
            foreach (var piece in Bpe(pieces))
                ids.Add(_vocab.TryGetValue(piece, out int id) ? id : EndOfTextId);
        }
        return ids;
    }

    public string Decode(IEnumerable<long> ids)
    {
        var chars = new StringBuilder();
        foreach (long id in ids)
        {
            if (id == EndOfTextId) continue;
            if (_idToToken.TryGetValue((int)id, out string? token))
                chars.Append(token);
        }

        var bytes = new List<byte>(chars.Length);
        foreach (char c in chars.ToString())
            if (_charToByte.TryGetValue(c, out byte b)) bytes.Add(b);
        return Encoding.UTF8.GetString(bytes.ToArray()).Trim();
    }

    private IEnumerable<string> Bpe(List<string> pieces)
    {
        while (pieces.Count > 1)
        {
            int best = int.MaxValue;
            (string A, string B)? selected = null;
            for (int i = 0; i + 1 < pieces.Count; i++)
            {
                if (_ranks.TryGetValue((pieces[i], pieces[i + 1]), out int rank) && rank < best)
                {
                    best = rank;
                    selected = (pieces[i], pieces[i + 1]);
                }
            }
            if (selected is null) break;

            var merged = new List<string>(pieces.Count);
            for (int i = 0; i < pieces.Count;)
            {
                if (i + 1 < pieces.Count && pieces[i] == selected.Value.A && pieces[i + 1] == selected.Value.B)
                {
                    merged.Add(pieces[i] + pieces[i + 1]);
                    i += 2;
                }
                else merged.Add(pieces[i++]);
            }
            pieces = merged;
        }
        return pieces;
    }

    private static (char[] ToChar, Dictionary<char, byte> ToByte) BuildByteMaps()
    {
        var bytes = new List<byte>();
        for (int i = 33; i <= 126; i++) bytes.Add((byte)i);
        for (int i = 161; i <= 172; i++) bytes.Add((byte)i);
        for (int i = 174; i <= 255; i++) bytes.Add((byte)i);

        var chars = new List<char>(bytes.Select(b => (char)b));
        int extra = 0;
        for (int i = 0; i < 256; i++)
        {
            if (bytes.Contains((byte)i)) continue;
            bytes.Add((byte)i);
            chars.Add((char)(256 + extra++));
        }

        var toChar = new char[256];
        var toByte = new Dictionary<char, byte>();
        for (int i = 0; i < bytes.Count; i++)
        {
            toChar[bytes[i]] = chars[i];
            toByte[chars[i]] = bytes[i];
        }
        return (toChar, toByte);
    }
}
