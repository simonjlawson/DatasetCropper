using System.IO;
using DatasetCropper.Services;
using Xunit;

namespace DatasetCropper.Tests;

public class TopazServiceTests
{
    [Theory]
    [InlineData(@"C:\Definitely\Not\Installed\tpai.exe")]
    [InlineData(@"C:\No\Such\Dir\tpai.exe")]
    public void Resolve_ConfiguredPathDoesNotExist_ReturnsNull(string configured)
    {
        string? result = TopazService.Resolve(configured);
        Assert.Null(result);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Resolve_EmptyOrWhitespace_UsesDefaultInstallPath(string configured)
    {
        // Empty/whitespace means "use the default install location". Whether
        // that resolves depends on whether Topaz is installed on this machine,
        // so assert the conditional contract rather than a fixed outcome.
        string? result = TopazService.Resolve(configured);
        string? expected = File.Exists(TopazService.DefaultTpaiPath) ? TopazService.DefaultTpaiPath : null;
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Resolve_ConfiguredPathIsRealFile_ReturnsThatPath()
    {
        string tmp = Path.Combine(Path.GetTempPath(), "DatasetCropper_topaz_" + Guid.NewGuid().ToString("N"), "tpai.exe");
        Directory.CreateDirectory(Path.GetDirectoryName(tmp)!);
        try
        {
            File.WriteAllText(tmp, "");
            string? result = TopazService.Resolve(tmp);
            Assert.Equal(tmp, result);
        }
        finally
        {
            try { Directory.Delete(Path.GetDirectoryName(tmp)!, true); } catch { /* best effort */ }
        }
    }

    // ---- Fake tpai + batch behavior ----

    /// <summary>
    /// Writes a fake tpai: a .cmd that delegates to a PowerShell script which
    /// copies every existing input file into the -o output directory.
    /// Returns the path of the .cmd (stands in for tpai.exe).
    /// </summary>
    private static string WriteFakeTpai(string dir)
    {
        string cmd = Path.Combine(dir, "fake-tpai.cmd");
        File.WriteAllText(cmd,
            "@echo off\r\n" +
            "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0fake-tpai.ps1\" %*\r\n");
        string ps1 = Path.Combine(dir, "fake-tpai.ps1");
        File.WriteAllText(ps1,
            "$outDir = $null\r\n" +
            "$copied = $false\r\n" +
            "for ($i = 0; $i -lt $args.Count; $i++) {\r\n" +
            "  $a = $args[$i]\r\n" +
            "  if ($a -eq '-o') { $outDir = $args[$i + 1]; $i++ }\r\n" +
            "  elseif ((Test-Path -LiteralPath $a) -and ($null -ne $outDir)) {\r\n" +
            "    Copy-Item -LiteralPath $a -Destination $outDir -Force\r\n" +
            "    $copied = $true\r\n" +
            "  }\r\n" +
            "}\r\n" +
            "if ($copied) { exit 0 } else { exit 255 }\r\n");
        return cmd;
    }

    [Fact]
    public void Upscale4xBatch_FakeTpai_MapsEveryInputToItsOutput()
    {
        string dir = Path.Combine(Path.GetTempPath(), "DatasetCropper_topaz_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string fake = WriteFakeTpai(dir);
            string in1 = Path.Combine(dir, "a.png");
            string in2 = Path.Combine(dir, "b.jpg");
            File.WriteAllBytes(in1, new byte[] { 1 });
            File.WriteAllBytes(in2, new byte[] { 2, 2 });
            string outDir = Path.Combine(dir, "out");

            var map = TopazService.Upscale4xBatch(fake, new[] { in1, in2 }, outDir, "--upscale");

            Assert.NotNull(map);
            Assert.Equal(2, map!.Count);
            Assert.True(File.Exists(map["a"]), "a.png must map to a produced file");
            Assert.True(File.Exists(map["b"]), "b.jpg must map to a produced file");
        }
        finally { try { Directory.Delete(dir, true); } catch { /* best effort */ } }
    }

    [Fact]
    public void Upscale4xBatch_NoOutputProduced_ReturnsEmptyNotNull()
    {
        // A fake tpai that produces nothing: the batch map must be empty
        // (not null) so the pipeline falls back per-image instead of failing.
        string dir = Path.Combine(Path.GetTempPath(), "DatasetCropper_topaz_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string fake = Path.Combine(dir, "noop.cmd");
            File.WriteAllText(fake, "@echo off\r\nexit /b 255\r\n");
            string in1 = Path.Combine(dir, "a.png");
            File.WriteAllBytes(in1, new byte[] { 1 });
            string outDir = Path.Combine(dir, "out");
            Directory.CreateDirectory(outDir);

            var map = TopazService.Upscale4xBatch(fake, new[] { in1 }, outDir, "--upscale");

            Assert.NotNull(map);
            Assert.Empty(map!);
        }
        finally { try { Directory.Delete(dir, true); } catch { /* best effort */ } }
    }
}