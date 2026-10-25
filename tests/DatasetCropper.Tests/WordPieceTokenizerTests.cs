using System.IO;
using DatasetCropper.Ml;
using Xunit;

namespace DatasetCropper.Tests;

public class WordPieceTokenizerTests : IDisposable
{
    // A small BERT-style vocab with the special tokens the tokenizer requires.
    //   "unhappy" is in the vocab whole, so it must NOT be split.
    //   "unhappiness" is NOT in the vocab, so it must split into "un" + "##happiness".
    private const string Json = """
        {
          "model": {
            "type": "WordPiece",
            "vocab": {
              "[PAD]": 0,
              "[UNK]": 100,
              "[CLS]": 101,
              "[SEP]": 102,
              "cat": 200,
              "dog": 300,
              ".": 400,
              "un": 700,
              "##happy": 701,
              "unhappy": 702,
              "##happiness": 705
            }
          }
        }
        """;

    private readonly string _path = Path.Combine(Path.GetTempPath(), "DatasetCropper_tok_" + Guid.NewGuid().ToString("N") + ".json");
    private readonly WordPieceTokenizer _tok;

    public WordPieceTokenizerTests()
    {
        File.WriteAllText(_path, Json);
        _tok = new WordPieceTokenizer(_path);
    }

    public void Dispose()
    {
        try { File.Delete(_path); } catch { /* best effort */ }
    }

    [Fact]
    public void Tokenize_AllTensorsHaveMaxLength()
    {
        var (ids, attn, types, pos, tmask) = _tok.Tokenize("cat");
        Assert.Equal(256, ids.Length);
        Assert.Equal(256, attn.Length);
        Assert.Equal(256, types.Length);
        Assert.Equal(256, pos.Length);
        Assert.Equal(256, tmask.Length);
    }

    [Fact]
    public void Tokenize_ClsFirst_SepAfterContent_PadAfter()
    {
        var (ids, attn, types, pos, _) = _tok.Tokenize("cat");
        Assert.Equal(101, ids[0]); // [CLS]

        // Content for "cat" is [CLS, cat, ., SEP] (a trailing '.' is appended).
        Assert.Equal(200, ids[1]); // cat
        Assert.Equal(400, ids[2]); // .
        Assert.Equal(102, ids[3]); // [SEP]

        // Everything after [SEP] is padding.
        Assert.Equal(0, ids[4]);   // [PAD]
        Assert.Equal(0, attn[4]);
        Assert.Equal(0, pos[4]);

        // attention_mask is 1 exactly for the real tokens, 0 for padding.
        Assert.Equal(1, attn[0]);
        Assert.Equal(1, attn[3]);
        Assert.Equal(0, attn[255]);

        // position_ids are 0..n-1 for real tokens.
        Assert.Equal(0, pos[0]);
        Assert.Equal(3, pos[3]);
    }

    [Fact]
    public void Tokenize_TextTokenMask_MarksOnlyWordTokens()
    {
        var (_, _, _, _, tmask) = _tok.Tokenize("cat");
        // [CLS]=0, cat=1, '.'=0 (separator, not a word), [SEP]=0.
        Assert.Equal(0, tmask[0]);
        Assert.Equal(1, tmask[1]);
        Assert.Equal(0, tmask[2]);
        Assert.Equal(0, tmask[3]);
    }

    [Fact]
    public void Tokenize_WholeWordInVocab_IsNotSplit()
    {
        var (ids, _, _, _, _) = _tok.Tokenize("unhappy");
        Assert.Equal(702, ids[1]); // "unhappy" matched whole, not "un"+"##happy"
    }

    [Fact]
    public void Tokenize_UnknownWord_SplitsByLongestMatch()
    {
        var (ids, _, _, _, _) = _tok.Tokenize("unhappiness");
        Assert.Equal(700, ids[1]); // "un"
        Assert.Equal(705, ids[2]); // "##happiness"
    }

    [Fact]
    public void Decode_ConcatenatesContinuationTokens()
    {
        Assert.Equal("unhappiness", _tok.Decode([700, 705])); // "un" + "##happiness"
        Assert.Equal("cathappy", _tok.Decode([200, 701]));     // "cat" + "##happy"
    }

    [Fact]
    public void Decode_JoinsSeparateTokensWithSpace()
    {
        Assert.Equal("cat dog", _tok.Decode([200, 300]));
    }

    [Fact]
    public void Decode_IgnoresUnknownIds()
    {
        Assert.Equal("cat", _tok.Decode([999, 200, 998]));
    }
}
