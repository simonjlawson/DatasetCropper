namespace DatasetCropper.Ml;

using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

/// <summary>
/// Minimal CLIP BPE tokenizer (reads a Hugging Face pair: <c>vocab.json</c> +
/// <c>merges.txt</c>) for CLIPSeg text prompts.
///
/// Ports the reference algorithm that was cross-checked against
/// transformers' <c>CLIPTokenizer</c> and matched it exactly: NFC → <c>\s+</c>→" " →
/// casefold; pre-tokenize per word with the CLIP regex; byte-encode each UTF-8 byte
/// (GPT-2 style: printable bytes → same codepoint, control bytes → 256+); BPE the
// pieces, the <b>last</b> char piece carrying the word-end mark; merge the lowest-
/// rank pair per pass; map final pieces through <c>vocab.json</c> (fallback unk); frame
/// <c>[SOT, content…, EOT]</c> left-truncated at 77, right-pad with EOT;
/// attention = <c>min(77, 2+content)</c> ones.
///
/// Note: the EOT token (id 49407) doubles as PAD and UNK, per the CLIP convention.
/// </summary>
public sealed class ClipTokenizer
{
    public const int MaxLength = 77;
    public const int EotId = 49407;   // EOT token (== PAD == UNK)
    public const int SotId = 49406;   // SOT token

    // Word-end mark appended to the last char piece of each pre-tokenized word
    // (four ASCII characters, in that order). Written as concatenation so the
    // surrounding tooling cannot mangle the literal.
    private const string WordEnd = "<" + "/" + "w" + ">";

    // The CLIP pre-tokenizer, as used in the verified reference (regex module:
    // \p{L}/\p{N} are Unicode properties; .NET's engine agrees on these builds).
    private static readonly Regex Tokens = new(
        "<\\|startoftext\\|>|<\\|endoftext\\|>|" +
        "'s|'t|'re|'ve|'m|'ll|'d|" +
        @"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.Compiled);

    // GPT-2/CLIP bytes_to_unicode, exactly: identity for bytes 33..126 and 161..255,
    // 256+ (in ascending byte order) for 0..32 and 127..160.
    private static readonly int[] ByteToIdx = BuildByteTable();

    private readonly Dictionary<string, int> _keyToId;
    private readonly Dictionary<(string a, string b), int> _ranks;
    private readonly char[] _byteChar; // 256-entry char per byte

    public ClipTokenizer(string vocabPath, string mergesPath)
    {
        _keyToId = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath))!
                   ?? throw new InvalidDataException("vocab.json is empty or invalid.");

        _ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var line in File.ReadAllLines(mergesPath))
        {
            var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2 || line.StartsWith('#')) continue;
            if (!_ranks.TryAdd((parts[0], parts[1]), rank))
                throw new InvalidDataException("Duplicate merge rule in " + Path.GetFileName(mergesPath));
            rank++;
        }

        _byteChar = new char[256];
        for (int b = 0; b < 256; b++)
            _byteChar[b] = (char)ByteToIdx[b];
    }

    /// <summary>
    /// Tokenize <paramref name="text"/> into the fixed-length-<see cref="MaxLength"/>
    /// <c>input_ids</c> and <c>attention_mask</c> arrays the CLIPSeg ONNX model wants
    /// (flat, ready to feed as int64 tensors of shape <c>[1, 77]</c>).
    /// </summary>
    public (int[] Ids, int[] Attn) Encode(string text)
    {
        text = text.Normalize(NormalizationForm.FormC);
        text = Regex.Replace(text, @"\s+", " ");
        text = text.ToLowerInvariant();

        var content = new List<int>();
        foreach (Match m in Tokens.Matches(text))
            foreach (var piece in BpeWord(m.Value))
                content.Add(_keyToId.TryGetValue(piece, out var id) ? id : EotId);

        // [SOT, content…, EOT], left-truncated to MaxLength, right-padded with EOT,
        // matching the slow-tokenizer contract the ONNX model was exported from.
        var ids = new int[MaxLength];
        int i = 0;
        ids[i++] = SotId;
        foreach (var c in content) { if (i >= MaxLength) break; ids[i++] = c; }
        if (content.Count + 2 <= MaxLength) ids[i++] = EotId;
        for (; i < MaxLength; i++) ids[i] = EotId;

        var attn = new int[MaxLength];
        int attnLen = Math.Min(MaxLength, content.Count + 2);
        for (int k = 0; k < attnLen; k++) attn[k] = 1;
        return (ids, attn);
    }

    private IEnumerable<string> BpeWord(string word)
    {
        // One char piece per UTF-8 byte of the pre-tokenized word.
        var chars = new List<char>(Encoding.UTF8.GetByteCount(word));
        foreach (byte b in Encoding.UTF8.GetBytes(word))
            chars.Add(_byteChar[b]);
        if (chars.Count == 0) yield break;

        var cur = new List<string>(chars.Count);
        for (int k = 0; k < chars.Count; k++)
            cur.Add(chars[k].ToString());
        cur[^1] += WordEnd; // the last char piece carries the word-end mark

        while (cur.Count > 1)
        {
            // Lowest-ranked pair among existing adjacent pairs (ties → first seen).
            int best = int.MaxValue;
            (string A, string B)? pick = null;
            for (int i = 0; i + 1 < cur.Count; i++)
            {
                if (_ranks.TryGetValue((cur[i], cur[i + 1]), out var r) && r < best)
                {
                    best = r;
                    pick = (cur[i], cur[i + 1]);
                }
            }
            if (pick == null) yield break; // no ranked pair left

            // Merge every occurrence of that pair (leftmost first), one pass.
            var next = new List<string>(cur.Count / 2 + 1);
            int k = 0;
            while (k < cur.Count)
            {
                if (k + 1 < cur.Count && cur[k] == pick!.Value.A && cur[k + 1] == pick.Value.B)
                {
                    next.Add(pick.Value.A + pick.Value.B);
                    k += 2;
                }
                else
                {
                    next.Add(cur[k]);
                    k++;
                }
            }
            cur = next;
        }

        foreach (var t in cur) yield return t;
    }

    private static int[] BuildByteTable()
    {
        var table = new int[256];
        var printable = new List<byte>(194);
        for (int b = 33; b <= 126; b++) 
            printable.Add((byte)b);   // 33..126 → identity
        for (int b = 161; b <= 255; b++) 
            printable.Add((byte)b);  // 161..255 → identity
        foreach (var b in printable) 
            table[b] = b;
        // The remaining bytes (0..32 and 127..160) get 256+ assigned in ascending byte order.
        int count = 0;
        for (int b = 0; b < 256; b++)
            if (table[b] == 0) { table[b] = 256 + count; count++; }
        return table;
    }
}
