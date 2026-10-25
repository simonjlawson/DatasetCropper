using System.Diagnostics;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DatasetCropper.ViewModels;
using Xunit;

namespace DatasetCropper.Tests;

public class OutputViewModelTests : IDisposable
{
    private readonly string _root;
    private readonly string _second;

    public OutputViewModelTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "DatasetCropper-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_root, "alpha"));
        _second = Path.Combine(_root, "beta");
        Directory.CreateDirectory(_second);
    }

    public void Dispose()
    {
        // Delete bottom-up; files may be briefly locked, so retry a few times.
        if (!Directory.Exists(_root)) return;
        for (int i = 0; i < 5; i++)
        {
            try
            {
                Directory.Delete(_root, recursive: true);
                return;
            }
            catch (IOException)
            {
                Thread.Sleep(200 * (i + 1));
            }
        }
    }

    /// <summary>Poll until <paramref name="path"/> can be opened exclusively, i.e. thumbnail generation has finished reading it.</summary>
    private static void WaitUntilUnlocked(string path)
    {
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 5000)
        {
            try
            {
                using (File.Open(path, FileMode.Open, FileAccess.Read, FileShare.None))
                {
                    return;
                }
            }
            catch (FileNotFoundException) { return; } // already gone — nothing to wait for
            catch (IOException) { Thread.Sleep(50); }
        }
    }

    [Fact]
    public void MoveSelectedTo_TargetFolder_MovesImageMaskAndCaption()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "a.png"));
        File.WriteAllBytes(Path.Combine(_root, "alpha", "a-masklabel.png"), new byte[] { 1 });
        File.WriteAllText(Path.Combine(_root, "alpha", "a.txt"), "tags");

        using var vm = new OutputViewModel(() => _root);
        vm.Refresh();
        WaitUntilUnlocked(src);
        var item = Assert.Single(vm.Items, i => i.RelativePath == "alpha/a.png");
        vm.Selection.Add(item);

        vm.MoveSelectedTo(2); // no concepts configured -> 2nd on-disk folder (alphabetical) = beta

        Assert.False(File.Exists(src));
        Assert.True(File.Exists(Path.Combine(_second, "a.png")));
        Assert.True(File.Exists(Path.Combine(_second, "a-masklabel.png")));
        Assert.Equal("tags", File.ReadAllText(Path.Combine(_second, "a.txt")));
        Assert.Contains("Moved 1 image", vm.Status);
        Assert.Contains("'beta'", vm.Status);
        Assert.Empty(vm.Selection); // single image was handled — grid advances, nothing left to select
    }

    [Fact]
    public void MoveSelectedTo_NameClash_SkipsFile()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "a.png"));
        WritePng(Path.Combine(_second, "a.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "beta" });
        vm.Refresh();
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/a.png"));

        vm.MoveSelectedTo(2);

        Assert.True(File.Exists(src)); // still in place
        Assert.Contains("skipped", vm.Status);
    }

    [Fact]
    public void MoveSelectedTo_AlreadyInTarget_NoChange()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "a.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "beta" });
        vm.Refresh();
        var item = vm.Items.First(i => i.RelativePath == "alpha/a.png");
        vm.Selection.Add(item);

        vm.MoveSelectedTo(1); // 1st concept = alpha — already there

        Assert.True(File.Exists(src));
        Assert.Contains("already", vm.Status, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(vm.Selection); // confirmed image was handled — advanced past it, nothing left to select
    }

    [Fact]
    public void MoveSelectedTo_AdvancesSelectionToNextImage()
    {
        WritePng(Path.Combine(_root, "alpha", "1.png"));
        string src = WritePng(Path.Combine(_root, "alpha", "2.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "gamma" });
        vm.Refresh();
        WaitUntilUnlocked(src);
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/2.png"));

        vm.MoveSelectedTo(2); // 2nd concept = gamma

        Assert.True(File.Exists(Path.Combine(_root, "gamma", "2.png")));
        var sel = Assert.Single(vm.Selection);
        Assert.Equal("alpha/1.png", sel.RelativePath); // advanced to the next unhandled image
    }

    [Fact]
    public void MoveSelectedTo_AdvancesSelectionWrapsAroundAtTail()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "1.png"));
        WritePng(Path.Combine(_root, "alpha", "2.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "gamma" });
        vm.Refresh();
        WaitUntilUnlocked(src);
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/1.png"));

        vm.MoveSelectedTo(2); // handled image is first — next unhandled wraps around

        Assert.True(File.Exists(Path.Combine(_root, "gamma", "1.png")));
        var sel = Assert.Single(vm.Selection);
        Assert.Equal("alpha/2.png", sel.RelativePath);
    }

    [Fact]
    public void ConfirmAndAdvance_AdvancesWithoutMovingFiles()
    {
        WritePng(Path.Combine(_root, "alpha", "1.png"));
        string second = WritePng(Path.Combine(_root, "alpha", "2.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha" });
        vm.Refresh();
        WaitUntilUnlocked(second);
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/1.png"));

        vm.ConfirmAndAdvanceCommand.Execute(null);

        Assert.True(File.Exists(Path.Combine(_root, "alpha", "1.png"))); // no move
        Assert.True(File.Exists(Path.Combine(_root, "alpha", "2.png")));
        var sel = Assert.Single(vm.Selection);
        Assert.Equal("alpha/2.png", sel.RelativePath);
        Assert.Contains("Confirmed", vm.Status, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ConfirmAndAdvance_WrapsSelectionAroundAtTail()
    {
        string first = WritePng(Path.Combine(_root, "alpha", "1.png"));
        WritePng(Path.Combine(_root, "alpha", "2.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha" });
        vm.Refresh();
        WaitUntilUnlocked(first);
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/2.png"));

        vm.ConfirmAndAdvanceCommand.Execute(null);

        Assert.True(File.Exists(Path.Combine(_root, "alpha", "2.png"))); // no move
        var sel = Assert.Single(vm.Selection);
        Assert.Equal("alpha/1.png", sel.RelativePath);
    }

    [Fact]
    public void MoveSelectedTo_BeyondFolderCount_NoChange()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "a.png"));

        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "beta" });
        vm.Refresh();
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/a.png"));

        vm.MoveSelectedTo(3); // only two concepts configured

        Assert.True(File.Exists(src));
        Assert.Contains("No concept 3", vm.Status);
    }

    [Fact]
    public void MoveSelectedTo_ConceptFolderMissing_CreatesIt()
    {
        string src = WritePng(Path.Combine(_root, "alpha", "a.png"));

        // 'gamma' is configured as a concept but has no output folder yet.
        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "gamma" });
        vm.Refresh();
        WaitUntilUnlocked(src);
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/a.png"));

        vm.MoveSelectedTo(2); // 2nd concept = gamma

        string dest = Path.Combine(_root, "gamma", "a.png");
        Assert.True(Directory.Exists(Path.Combine(_root, "gamma")));
        Assert.False(File.Exists(src));
        Assert.True(File.Exists(dest));
        Assert.Contains("Moved 1 image", vm.Status);
        Assert.Contains("'gamma'", vm.Status);
    }

    [Fact]
    public void DeleteSelected_SelectsNextSurvivorForPreview()
    {
        WritePng(Path.Combine(_root, "alpha", "1.png"));
        WritePng(Path.Combine(_second, "1.png"));
        WritePng(Path.Combine(_second, "2.png"));

        using var vm = new OutputViewModel(() => _root);
        vm.Refresh();
        WaitUntilUnlocked(Path.Combine(_root, "alpha", "1.png"));
        vm.Selection.Add(vm.Items.First(i => i.RelativePath == "alpha/1.png"));

        vm.DeleteSelectedCommand.Execute(null);

        Assert.False(File.Exists(Path.Combine(_root, "alpha", "1.png")));
        var sel = Assert.Single(vm.Selection);
        Assert.Equal("beta/1.png", sel.RelativePath);
    }

    [Fact]
    public void FolderHint_ShowsNumberedConcepts()
    {
        using var vm = new OutputViewModel(() => _root, () => new[] { "alpha", "beta" });
        vm.Refresh();
        Assert.Equal("1=alpha  2=beta", vm.FolderHint);
    }

    [Fact]
    public void FolderHint_FallsBackToOnDiskFolders_WhenNoConcepts()
    {
        using var vm = new OutputViewModel(() => _root); // no concepts supplier
        vm.Refresh();
        Assert.Equal("1=alpha  2=beta", vm.FolderHint); // from the folders in the fixture
    }

    [Fact]
    public void FolderHint_SingleFolderAndMultiple()
    {
        var vm = new OutputViewModel(() => _root);
        Assert.Equal("1=alpha", OutputViewModel.BuildFolderHint(new[] { "alpha" }));
        Assert.Equal("1=alpha  2=beta", OutputViewModel.BuildFolderHint(new[] { "alpha", "beta" }));
        Assert.Equal(string.Empty, OutputViewModel.BuildFolderHint(Array.Empty<string>()));
        vm.Dispose();
    }

    /// <summary>A real decodable PNG so thumbnail/preview loading in headless tests succeeds.</summary>
    private static string WritePng(string path)
    {
        using var img = new Image<Rgba32>(8, 8);
        img.Mutate(x => x.BackgroundColor(Color.FromPixel(new Rgba32(200, 30, 90, 255))));
        img.Save(path);
        return path;
    }
}
