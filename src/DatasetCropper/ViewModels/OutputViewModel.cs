using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DatasetCropper.Imaging;

namespace DatasetCropper.ViewModels;

/// <summary>
/// Backs the Output tab: a thumbnail grid of every crop in the output folder
/// (all subdirectories), a side-by-side full-size preview of the selection,
/// Del-key deletion of the selected image plus its -masklabel.png / .txt, and
/// number-key (1-9) assignment of the selection to the Nth configured concept's
/// folder (Composition list, then Subject list; on-disk folders when none are set).
/// Enter confirms the selection's current concept. After handling an image the
/// grid advances to the next one (wrapping around) so triage flows image to image.
/// Triaged (confirmed or reassigned) images are hidden from the grid until the folder
/// filter is set to All or the grid is refreshed.
/// </summary>
public partial class OutputViewModel : ObservableObject, IDisposable
{
    private const int ThumbnailSize = 256;
    private const int SaveDebounceMs = 400;
    private const string AllFolder = "All";

    private readonly Func<string> _outputFolder;
    private readonly Func<IReadOnlyList<string>>? _concepts;
    private CancellationTokenSource? _thumbCts;
    private CancellationTokenSource? _previewCts;
    private string _filter = AllFolder;
    private readonly HashSet<string> _triaged = new(StringComparer.OrdinalIgnoreCase);
    private bool _refreshing;
    private bool _converting;

    public ObservableCollection<OutputItemViewModel> Items { get; } = new();

    /// <summary>Two-way synced from the grid's selection (extended selection).</summary>
    public ObservableCollection<OutputItemViewModel> Selection { get; } = new();

    /// <summary>Raised (UI thread) after Refresh with an item the UI should select in the grid.</summary>
    public event EventHandler<OutputItemViewModel>? SelectItemRequested;

    /// <summary>Folders in the filter combo: one per top-level output folder, plus the All sentinel.</summary>
    public ObservableCollection<string> Filters { get; } = new() { AllFolder };

    /// <summary>The top-level output folder shown in the grid (All = every folder). Triaged images
    /// (confirmed or reassigned) stay hidden until All is selected again or the grid is refreshed.</summary>
    public string Filter
    {
        get => _filter;
        set
        {
            // The combo may transiently clear SelectedItem while it rebuilds items - treat null as All.
            var target = value is null ? AllFolder : value;
            if (string.Equals(_filter, target, StringComparison.OrdinalIgnoreCase))
                return;
            _filter = target;
            if (_refreshing == false)
                Refresh();
        }
    }

    [ObservableProperty]
    private bool _canEditCaption;

    [ObservableProperty]
    private string _selectedConcept = string.Empty;

    [ObservableProperty]
    private int _jpegQualityLevel = 2;

    public string JpegQualityDescription => JpegQualityLevel switch
    {
        0 => $"Good ({JpgConversion.GoodQuality}%)",
        1 => $"Better ({JpgConversion.BetterQuality}%)",
        _ => $"Best ({JpgConversion.JpegQuality}%)"
    };

    partial void OnJpegQualityLevelChanged(int value)
        => OnPropertyChanged(nameof(JpegQualityDescription));

    private string _caption = string.Empty;
    /// <summary>Editable caption — the contents of the selection's sibling .txt, auto-saved on edit.</summary>
    public string Caption
    {
        get => _caption;
        set
        {
            if (value == _caption) return;
            _caption = value;
            OnPropertyChanged(nameof(Caption));
            if (!_loadingCaption) ScheduleSave();
        }
    }

    private string?[]? _pendingSelectionPaths;
    private string? _savePath;
    private int _saveSeq;
    private bool _loadingCaption;
    private CancellationTokenSource? _captionCts;

    [ObservableProperty]
    private bool _hasItems;

    [ObservableProperty]
    private ImageSource? _preview;

    [ObservableProperty]
    private string _status = "No output folder set.";

    /// <summary>The 1-9 mapping of the output folders (e.g. "1=cat  2=dog"), shown next to the buttons.</summary>
    [ObservableProperty]
    private string _folderHint = string.Empty;

    /// <param name="outputFolder">Supplies the folder to browse.</param>
    /// <param name="concepts">Supplies the numbered assignment targets — the configured
    /// composition concepts followed by the subject concepts, in order. <c>null</c> falls back
    /// to the folders that exist on disk (alphabetical).</param>
    public OutputViewModel(Func<string> outputFolder, Func<IReadOnlyList<string>>? concepts = null)
    {
        _outputFolder = outputFolder;
        _concepts = concepts;
    }

    public void Dispose()
    {
        _thumbCts?.Cancel();
        _previewCts?.Cancel();
        _captionCts?.Cancel();
    }

    /// <summary>
    /// Re-scans the output folder and rebuilds the grid. Called when the tab is
    /// shown, after a run finishes, after the output folder changes, and after
    /// a deletion.
    /// </summary>
    public void Refresh()
    {
        _thumbCts?.Cancel();
        _thumbCts = new CancellationTokenSource();
        CancellationToken token = _thumbCts.Token;

        string root = _outputFolder();
        Items.Clear();
        Selection.Clear();
        Preview = null;
        HasItems = false;

        if (string.IsNullOrWhiteSpace(root))
        {
            FolderHint = string.Empty;
            Status = "No output folder set.";
            return;
        }
        if (!Directory.Exists(root))
        {
            FolderHint = string.Empty;
            Status = $"Output folder does not exist yet: {root}";
            return;
        }

        // Rebuild the filter list from folders that exist, keeping the current choice if it survived.
        _refreshing = true;
        try
        {
            var folderNames = Directory.EnumerateDirectories(root)
                .Select(Path.GetFileName)
                .Where(n => n is { Length: > 0 } && n[0] != '.')
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                .ToList();

            foreach (var existing in Filters.Where(f => f != AllFolder).ToList())
                if (folderNames.Contains(existing) == false)
                    Filters.Remove(existing);

            foreach (var folder in folderNames)
                if (Filters.Contains(folder) == false)
                    Filters.Add(folder);

            if (string.Equals(_filter, AllFolder, StringComparison.OrdinalIgnoreCase) == false
                && Filters.Contains(_filter) == false)
                _filter = AllFolder;
        }
        finally
        {
            _refreshing = false;
            OnPropertyChanged(nameof(Filter));
        }

        bool showAll = string.Equals(_filter, AllFolder, StringComparison.OrdinalIgnoreCase);
        // Crops may be PNG or JPG (the Convert-to-JPG option re-encodes them); masks stay PNG.
        var files = Directory.EnumerateFiles(root, "*.png", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(root, "*.jpg", SearchOption.AllDirectories))
            .Where(f => !f.EndsWith("-masklabel.png", StringComparison.OrdinalIgnoreCase))
            .Where(f => showAll || (string.Equals(TopFolder(root, f), _filter, StringComparison.OrdinalIgnoreCase)
                && _triaged.Contains(f) == false))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var f in files)
        {
            string rel = Path.GetRelativePath(root, f).Replace('\\', '/');
            Items.Add(new OutputItemViewModel(f, rel));
        }

        Status = showAll
            ? $"{Items.Count} image(s) in {root}"
            : $"{Items.Count} image(s) in {root} ({_filter})";
        HasItems = Items.Count > 0;
        FolderHint = BuildFolderHint(ResolveTargets(root, _concepts));
        _ = GenerateThumbnailsAsync(token);

        // Re-select after a delete/move: the first match drives the grid's
        // selection (via the event), the rest join the VM-level selection.
        var targets = _pendingSelectionPaths;
        _pendingSelectionPaths = null;
        if (targets is { Length: > 0 })
        {
            Selection.Clear();
            bool flagged = false;
            foreach (string target in targets)
            {
                var match = Items.FirstOrDefault(i =>
                    string.Equals(i.FilePath, target, StringComparison.OrdinalIgnoreCase));
                if (match is null) continue;
                if (!flagged) { SelectItemRequested?.Invoke(this, match); flagged = true; }
                Selection.Add(match);
            }
            if (Selection.Count > 0) OnSelectionChanged();
        }
    }

    /// <summary>
    /// Mirrors the grid's current selection into <see cref="Selection"/> and
    /// refreshes the preview. Called from the ListBox's SelectionChanged.
    /// </summary>
    public void SyncSelection(IEnumerable<OutputItemViewModel> selected)
    {
        var list = selected.ToList();
        bool changed = list.Count != Selection.Count || list.Any(i => !Selection.Contains(i));
        if (!changed) return;

        Selection.Clear();
        foreach (var item in list)
            Selection.Add(item);
        OnSelectionChanged();
    }

    [RelayCommand]
    private void DeleteSelected()
    {
        if (Selection.Count == 0) return;

        var toDelete = Selection.ToList();

        // After the refresh, select the surviving image that follows the
        // earliest deleted one (or the last survivor when it was the tail).
        int firstIdx = toDelete.Min(i => Items.IndexOf(i));
        var survivor = Items.Skip(firstIdx + 1).FirstOrDefault(i => !toDelete.Contains(i))
            ?? Items.Where(i => !toDelete.Contains(i)).LastOrDefault();
        _pendingSelectionPaths = survivor is null
            ? null
            : new[] { survivor.FilePath };

        int filesDeleted = 0;
        foreach (var item in toDelete)
        {
            string dir = Path.GetDirectoryName(item.FilePath) ?? "";
            string stem = Path.GetFileNameWithoutExtension(item.FilePath);
            foreach (var f in new[]
                     {
                         item.FilePath,
                         Path.Combine(dir, stem + "-masklabel.png"),
                         Path.Combine(dir, stem + ".txt"),
                     })
            {
                try
                {
                    if (File.Exists(f)) { File.Delete(f); filesDeleted++; }
                }
                catch { /* locked or vanished — skip */ }
            }
        }

        Refresh();
        Status = $"Deleted {toDelete.Count} image(s), {filesDeleted} file(s) removed.";
    }

    /// <summary>
    /// Moves the selection — each image plus its sibling -masklabel.png / .txt — to the
    /// folder of the Nth configured concept (1-based; Composition list, then Subject
    /// list): the folder is created if it doesn't exist yet. After the handled
    /// images the grid advances to the next unhandled one (wrapping around) — the
    /// same next-image flow as confirming with Enter or deleting.
    /// </summary>
    public void MoveSelectedTo(int folderNumber)
    {
        if (Selection.Count == 0) return;

        string root = _outputFolder();
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root))
        {
            Status = "No output folder to move within.";
            return;
        }

        var assignments = ResolveTargets(root, _concepts);
        if (assignments.Count == 0)
        {
            Status = "No concepts configured — add compositions on the Process tab.";
            return;
        }
        if (folderNumber < 1 || folderNumber > assignments.Count)
        {
            Status = $"No concept {folderNumber} — available: {BuildFolderHint(assignments)}.";
            return;
        }

        string target = assignments[folderNumber - 1];
        string destDir = Path.Combine(root, target);
        var toMove = Selection.ToList();
        var targets = new List<string>(toMove.Count);
        var settled = new List<OutputItemViewModel>(toMove.Count);
        int moved = 0, skipped = 0, already = 0;

        foreach (var item in toMove)
        {
            string srcFile = item.FilePath;
            string dir = Path.GetDirectoryName(srcFile) ?? "";
            string rel = Path.GetRelativePath(root, dir).Replace('\\', '/');
            if (rel == target)
            {
                // Reassigning to the current folder is still a confirmed choice - mark it handled.
                already++;
                settled.Add(item);
                _triaged.Add(srcFile);
                continue;
            }

            string destFile = Path.Combine(destDir, Path.GetFileName(srcFile));
            if (File.Exists(destFile)) { skipped++; continue; } // name clash — leave in place

            try
            {
                Directory.CreateDirectory(destDir);
                MoveWithRetry(srcFile, destFile);
                string stem = Path.GetFileNameWithoutExtension(srcFile);
                foreach (var suffix in new[] { "-masklabel.png", ".txt" })
                {
                    string sibling = Path.Combine(dir, stem + suffix);
                    if (File.Exists(sibling))
                        MoveWithRetry(sibling, Path.Combine(destDir, stem + suffix));
                }
                targets.Add(destFile);
                settled.Add(item);
                moved++;
                _triaged.Add(destFile); // triaged - assigned to its chosen folder
            }
            catch { skipped++; } // locked or vanished — leave in place
        }

        if (settled.Count > 0)
        {
            // Triaged (moved, or already in the target) — advance the grid to the
            // next unhandled image so the user can handle that one next.
            _pendingSelectionPaths = NextPendingPath(settled);
            Refresh();
        }
        Status = moved == 0
            ? (already > 0
                ? $"Already in '{target}'."
                : $"Move to '{target}': {skipped} image(s) skipped (name clash or locked).")
            : $"Moved {moved} image(s) to '{target}'{(skipped > 0 ? $", {skipped} skipped" : "")}{(already > 0 ? $", {already} already there" : "")}.";
    }

    /// <summary>The move source can be briefly locked by a preview or thumbnail load; wait a moment for the handle to be released before giving up.</summary>
    private static void MoveWithRetry(string source, string destination)
    {
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                File.Move(source, destination);
                return;
            }
            catch (IOException) when (attempt < 5)
            {
                Thread.Sleep(100 * (attempt + 1));
            }
        }
    }

    /// <summary>
    /// Enter: the selection's current concept is correct — no files move. The grid
    /// advances to the next image (wrapping around), the same next-image flow as
    /// assignment and deletion.
    /// </summary>
    [RelayCommand]
    private void ConfirmAndAdvance()
    {
        if (Selection.Count == 0) { Status = "Nothing selected."; return; }

        var handled = new HashSet<OutputItemViewModel>(Selection);
        var next = NextAfter(handled);

        // Triaged: the confirmed image(s) leave the grid; the folder filter is re-applied and
        // the next image is re-selected through the same flow as move and delete.
        foreach (var item in handled)
            _triaged.Add(item.FilePath);

        Selection.Clear();
        _pendingSelectionPaths = next is null ? null : new[] { next.FilePath };
        Refresh();
        Status = next is null
            ? "Confirmed — that was the only image in the grid."
            : "Confirmed the current concept — advanced to the next image.";
    }

    /// <summary>Restores the grid as it was: cleared triaged (confirmed or reassigned) images and
    /// sets the folder filter back to All.</summary>
    [RelayCommand]
    public void RefreshAll()
    {
        _triaged.Clear();
        _filter = AllFolder;
        OnPropertyChanged(nameof(Filter));
        Refresh();
        Status = $"Grid restored — {Items.Count} image(s).";
    }

    /// <summary>
    /// Converts every output crop (PNG) under the output folder to a same-stem JPG
    /// in place — <c>*-masklabel.png</c> files are left as PNG. Runs on a background
    /// thread; refreshes the grid when done.
    /// </summary>
    [RelayCommand]
    private void ConvertToJpg()
    {
        string root = _outputFolder();
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root))
        {
            Status = "No output folder to convert.";
            return;
        }
        if (_converting) return;
        _converting = true;
        Status = "Converting PNG crops to JPG…";
        int quality = JpgConversion.QualityForLevel(JpegQualityLevel);

        _ = Task.Run(() =>
        {
            bool crashed = false;
            int converted = 0, skipped = 0, failed = 0;
            try
            {
                (converted, skipped, failed) = JpgConversion.ConvertPngs(root, quality);
            }
            catch { crashed = true; }
            finally
            {
                _converting = false;
            }

            Application.Current?.Dispatcher.Invoke(() =>
            {
                Refresh();
                Status = crashed
                    ? "Conversion failed — could not read the output folder."
                    : $"Converted {converted} image(s) to JPG{(skipped > 0 ? $", {skipped} skipped" : "")}{(failed > 0 ? $", {failed} failed" : "")} — masks stay PNG.";
            });
        });
    }



    /// <summary>The top-level (concept/subject) folder a file lives in; empty for root files and hidden folders.</summary>
    private static string TopFolder(string root, string file)
    {
        var rel = Path.GetRelativePath(root, file).Replace('\\', '/');
        int sep = rel.IndexOf('/');
        string name = sep > 0 ? rel[..sep] : string.Empty;
        return name.Length > 0 && name[0] != '.' ? name : string.Empty;
    }



    /// <summary>Next unhandled image after the earliest handled one (wrapping around); null when all are handled.</summary>
    private OutputItemViewModel? NextAfter(HashSet<OutputItemViewModel> handled)
    {
        int first = handled.Min(i => Items.IndexOf(i));
        return Items.Skip(first + 1).FirstOrDefault(i => !handled.Contains(i))
            ?? Items.FirstOrDefault(i => !handled.Contains(i));
    }

    /// <summary>Pending selection path after a handled image, for the next-image advance (null when none left).</summary>
    private string?[]? NextPendingPath(IEnumerable<OutputItemViewModel> handled)
    {
        var next = NextAfter(new HashSet<OutputItemViewModel>(handled));
        return next is null ? null : new[] { next.FilePath };
    }

    /// <summary>The 1-9 mapping of the output folders, e.g. "1=alpha  2=beta".</summary>
    public static string BuildFolderHint(IReadOnlyList<string> folders)
    {
        if (folders.Count == 0) return string.Empty;
        var parts = new List<string>();
        for (int i = 0; i < folders.Count && i < 9; i++)
            parts.Add((i + 1) + "=" + folders[i]);
        if (folders.Count > 9) parts.Add("+more");
        return string.Join("  ", parts);
    }

    /// <summary>
    /// The numbered assignment targets, in order: the configured concepts
    /// (composition, then subjects), sanitized to folder names and deduped;
    /// falls back to the folders on disk when concept names resolve to none.
    /// </summary>
    private static List<string> ResolveTargets(string root, Func<IReadOnlyList<string>>? concepts)
    {
        if (concepts is not null)
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var list = new List<string>();
            foreach (var raw in concepts())
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                string name = Services.ProcessingService.SanitizeFolder(raw);
                if (name == Services.ProcessingService.UnsortedFolder) continue; // blank concept
                if (seen.Add(name)) list.Add(name);
            }
            if (list.Count > 0) return list;
        }
        return VisibleFolders(root);
    }

    /// <summary>The visible (non-hidden) output subfolders, alphabetical (the no-concepts fallback).</summary>
    private static List<string> VisibleFolders(string root)
    {
        try
        {
            return Directory.EnumerateDirectories(root)
                .Select(Path.GetFileName)
                .Where(n => n is { Length: > 0 } && n[0] != '.') // skip hidden folders like .topaz
                .Cast<string>()
                .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch { return new List<string>(); }
    }

    private void OnSelectionChanged()
    {
        LoadCaption();

        var first = Selection.FirstOrDefault();
        if (first is null)
        {
            Preview = null;
            return;
        }

        _previewCts?.Cancel();
        var cts = new CancellationTokenSource();
        _previewCts = cts;
        string path = first.FilePath;

        _ = Task.Run(() =>
        {
            try
            {
                using var img = Image.Load<Rgba32>(path);
                var src = ToBitmapSource(img);
                if (!cts.Token.IsCancellationRequested)
                    Application.Current?.Dispatcher.Invoke(() => Preview = src);
            }
            catch { /* keep the previous preview if the file is unreadable */ }
        });
    }

    /// <summary>
    /// Loads the selection's primary concept (its output subfolder) and the
    /// sibling .txt contents into the editable caption box.
    /// </summary>
    private void LoadCaption()
    {
        _loadingCaption = true;
        _saveSeq++;
        _captionCts?.Cancel();

        var first = Selection.FirstOrDefault();
        if (first is null)
        {
            _savePath = null;
            CanEditCaption = false;
            SelectedConcept = string.Empty;
            Caption = string.Empty;
            _loadingCaption = false;
            return;
        }

        string txtPath = Path.ChangeExtension(first.FilePath, ".txt");
        _savePath = txtPath;
        CanEditCaption = true;
        SelectedConcept = string.IsNullOrWhiteSpace(Path.GetDirectoryName(first.RelativePath))
            ? "(untagged)"
            : Path.GetDirectoryName(first.RelativePath);

        var cts = new CancellationTokenSource();
        _captionCts = cts;
        string text = string.Empty;
        if (File.Exists(txtPath))
        {
            try { text = File.ReadAllText(txtPath); }
            catch { /* keep empty if unreadable */ }
        }
        if (cts.Token.IsCancellationRequested) return;
        try
        {
            Application.Current?.Dispatcher.Invoke(() =>
            {
                if (!ReferenceEquals(_savePath, txtPath)) return;
                _loadingCaption = true;
                Caption = text;
                _loadingCaption = false;
            });
        }
        catch { /* window closed mid-load */ }
    }

    /// <summary>
    /// Debounced write of the caption back to the .txt file; the latest edit
    /// wins and stale writes are dropped.
    /// </summary>
    private void ScheduleSave()
    {
        if (_savePath is null) return;
        int seq = Interlocked.Increment(ref _saveSeq);
        string path = _savePath;
        string text = _caption;
        try
        {
            Task.Delay(SaveDebounceMs).ContinueWith(_ =>
            {
                if (seq != _saveSeq) return;
                try { File.WriteAllText(path, text); }
                catch { /* locked or vanished — skip */ }
            });
        }
        catch { /* OOM under load — nothing sensible to do */ }
    }

    private async Task GenerateThumbnailsAsync(CancellationToken token)
    {
        var pending = Items.ToList();
        using var throttle = new SemaphoreSlim(4);
        var tasks = pending.Select(async item =>
        {
            await throttle.WaitAsync(token);
            try
            {
                string path = item.FilePath;
                ImageSource? thumb = await Task.Run(() =>
                {
                    using var img = Image.Load<Rgba32>(path);
                    img.Mutate(x => x.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
                    {
                        Mode = SixLabors.ImageSharp.Processing.ResizeMode.Max,
                        Size = new SixLabors.ImageSharp.Size(ThumbnailSize, ThumbnailSize)
                    }));
                    return ToBitmapSource(img);
                }, token);
                if (!token.IsCancellationRequested)
                    Application.Current?.Dispatcher.Invoke(() => item.Thumbnail = thumb);
            }
            catch { /* leave the cell blank if the file is unreadable */ }
            finally { throttle.Release(); }
        });
        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Copies the (live) pixel buffer into a frozen, UI-thread-safe
    /// <see cref="BitmapSource"/>. ImageSharp stores Rgba32 as RGBA bytes in
    /// memory; WPF's Bgra32 format is non-premultiplied BGRA, so the channels
    /// are swapped while copying.
    /// </summary>
    internal static ImageSource ToBitmapSource(Image<Rgba32> img)
    {
        byte[] pixels = new byte[img.Width * img.Height * 4];
        int offset = 0;
        img.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < img.Height; y++)
            {
                var row = acc.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    var p = row[x];
                    pixels[offset++] = p.B;
                    pixels[offset++] = p.G;
                    pixels[offset++] = p.R;
                    pixels[offset++] = p.A;
                }
            }
        });
        var bmp = BitmapSource.Create(
            img.Width, img.Height, 96, 96,
            PixelFormats.Bgra32, null,
            pixels, img.Width * 4);
        bmp.Freeze();
        return bmp;
    }
}
