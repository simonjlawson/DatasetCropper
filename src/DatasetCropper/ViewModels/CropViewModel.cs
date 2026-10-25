using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DatasetCropper.ViewModels;

/// <summary>
/// Backs the Crop tab — a manual crop tool modelled on PixelPruner:
/// the source image is shown fit-to-pane (capped at native size), a fixed
/// square crop box (default 1024 px of the original image) is centered on the
/// mouse cursor and may straddle the image edge; the highlighted rectangle is
/// the in-image crop area and a live preview shows it; a click (or "Save crop")
/// writes that portion to
/// &lt;OutputFolder&gt;/manual_crops/cropped_N_&lt;stem&gt;.png, aspect-preserving,
/// scaled into the target size. Saved crops can then be added to the Process queue and run
/// through the pipeline as source images.
/// </summary>
public partial class CropViewModel : ObservableObject
{
    /// <summary>Longest side of the in-pane display bitmap (WPF scales it further if the pane is larger).</summary>
    private const int MaxDisplaySide = 1600;

    /// <summary>Preview-pane render size cap (PixelPruner caps its preview at 512).</summary>
    private const int MaxPreviewSide = 512;

    private const int ThumbnailSize = 256;

    /// <summary>Subfolder (under the output folder) that saved crops land in.</summary>
    public const string CropsFolderName = "manual_crops";

    private readonly Func<IReadOnlyList<string>> _imagePaths;
    private readonly Func<string> _outputFolder;
    private readonly Action<IEnumerable<string>> _queueCrops;

    private List<string> _images = [];
    private Image<Rgba32>? _current;       // full-resolution current image (kept for crisp crops — like PixelPruner's in-memory original)
    private int _nativeW, _nativeH;

    // Native-pixel crop box (top-left + side). Display values derive from these + the mapping below.
    private int _boxNx, _boxNy, _boxSide;
    private Rectangle? _lastPreview = null;

    // Container-DIP mapping for the current container size (set by Relayout/MoveBox).
    private double _scale = -1, _offX, _offY;

    private CancellationTokenSource? _loadCts;
    private CancellationTokenSource? _previewCts;
    private int _cropCounter;
    private bool _processingCropQueue;

    private sealed class PendingCrop
    {
        public required Image<Rgba32> Image { get; init; }
        public required int OutputWidth { get; init; }
        public required int OutputHeight { get; init; }
        public required string Stem { get; init; }
        public required string Folder { get; init; }
    }

    private readonly Queue<PendingCrop> _pendingCrops = new();

    /// <summary>1:1 preset sizes offered in the tab's combo box.</summary>
    /// <summary>Crop size from the mouse wheel: step per notch and the range (px of the original image).</summary>
    private const int CropSizeWheelStep = 128;

    private const int CropSizeWheelMin = 1024;

    private const int CropSizeWheelMax = 16384;

    public int[] CropSizes { get; } = [512, 1024, 1536, 2048];

    public ObservableCollection<CropItemViewModel> Crops { get; } = new();

    [ObservableProperty]
    private bool _hasCrops;

    [ObservableProperty]
    private bool _hasImage;

    [ObservableProperty]
    private ImageSource? _imageSource;

    [ObservableProperty]
    private ImageSource? _cropPreview;

    [ObservableProperty]
    private string _imageName = "";

    [ObservableProperty]
    private string _imageSize = "";

    [ObservableProperty]
    private int _currentIndex;

    [ObservableProperty]
    private int _imageCount;

    /// <summary>"2 of 5" — recomputed whenever the index or count changes.</summary>
    public string PositionText => _imageCount == 0 ? string.Empty : $"{_currentIndex + 1} of {_imageCount}";

    /// <summary>True when at least one image is in the list (drives the Prev/Next enabled state).</summary>
    public bool CanNavigate => _imageCount > 0;

    // Crop-box overlay (container DIP coordinates).
    [ObservableProperty]
    private double _boxLeft;

    [ObservableProperty]
    private double _boxTop;

    [ObservableProperty]
    private double _boxWidth;

    [ObservableProperty]
    private double _boxHeight;

    [ObservableProperty]
    private bool _boxVisible;

    /// <summary>Target crop size in original-image pixels (1:1 — the saved file's square size when the box is fully inside the image; a clipped crop saves the in-image portion aspect-preserving).</summary>
    [ObservableProperty]
    private int _cropSize = 1024;

    /// <summary>
    /// Text bridge for the editable size combo: round-trips to <see cref="CropSize"/>;
    /// invalid input just falls back to the current size.
    /// </summary>
    public string? CropSizeText
    {
        get => CropSize.ToString();
        set
        {
            if (value is null || !int.TryParse(value.Trim(), out int sz)) return;
            CropSize = sz;
        }
    }

    [ObservableProperty]
    private string _status = "Add images (Process tab) or 'Open image...' to start cropping.";

    /// <param name="imagePaths">Current croppable images (the Process queue).</param>
    /// <param name="outputFolder">Folder saved crops land in (root of manual_crops).</param>
    /// <param name="queueCrops">Callback that adds saved crops to the Process queue.</param>
    public CropViewModel(Func<IReadOnlyList<string>> imagePaths, Func<string> outputFolder,
                         Action<IEnumerable<string>> queueCrops)
    {
        _imagePaths = imagePaths;
        _outputFolder = outputFolder;
        _queueCrops = queueCrops;
    }

    partial void OnCurrentIndexChanged(int value) => OnPropertyChanged(nameof(PositionText));

    partial void OnImageCountChanged(int value)
    {
        OnPropertyChanged(nameof(PositionText));
        OnPropertyChanged(nameof(CanNavigate));
    }

    partial void OnCropSizeChanged(int value)
    {
        OnPropertyChanged(nameof(CropSizeText));
        // Keep whatever point of the image the box was centred on, with the new side.
        if (_current is null || _boxSide == value) return;
        int cx = (int)Math.Round(_boxNx + _boxSide / 2.0);
        int cy = (int)Math.Round(_boxNy + _boxSide / 2.0);
        _boxSide = NativeSide();
        _boxNx = Math.Clamp(cx - _boxSide / 2, 1 - _boxSide, _nativeW - 1);
        _boxNy = Math.Clamp(cy - _boxSide / 2, 1 - _boxSide, _nativeH - 1);
        PlaceBox();
    }

    /// <summary>
    /// Re-syncs from the image list (Process queue) and (re)loads the current
    /// image. Called on tab load, after the queue changes, and after adding
    /// images from the tab itself.
    /// </summary>
    public void Refresh()
    {
        _images = _imagePaths().Where(File.Exists).ToList();
        // Write through the generated property — a bare field write skips the
        // PropertyChanged notification, which left the Prev/Next bindings frozen
        // at their initial (disabled) value.
        ImageCount = _images.Count;
        if (CurrentIndex >= _images.Count)
            CurrentIndex = 0;

        if (_imageCount == 0)
        {
            _loadCts?.Cancel();
            _current?.Dispose();
            _current = null;
            HasImage = false;
            ImageSource = null;
            CropPreview = null;
            ImageName = "";
            ImageSize = "";
            BoxVisible = false;
            return;
        }

        LoadCurrent();
    }

    /// <summary>Called by <see cref="MainViewModel"/> whenever the queue changes.</summary>
    public void OnQueueChanged() => Refresh();

    private void LoadCurrent()
    {
        string path = _images[_currentIndex];
        _loadCts?.Cancel();
        var cts = new CancellationTokenSource();
        _loadCts = cts;
        _lastPreview = null;

        Status = $"Loading {Path.GetFileName(path)}...";
        HasImage = false;
        ImageSource = null;
        CropPreview = null;
        BoxVisible = false;

        Task.Run(() =>
        {
            Image<Rgba32>? img = null;
            ImageSource? display = null;
            Exception? error = null;
            try
            {
                img = Image.Load<Rgba32>(path);
                int w = img.Width, h = img.Height;

                // Display bitmap: downscale for the pane (WPF scales it further if needed);
                // keep the full-resolution original in memory for the crops themselves.
                if (Math.Max(w, h) > MaxDisplaySide)
                {
                    using var small = img.Clone(ctx => ctx.Resize(new ResizeOptions
                    {
                        Mode = SixLabors.ImageSharp.Processing.ResizeMode.Max,
                        Size = new SixLabors.ImageSharp.Size(MaxDisplaySide, MaxDisplaySide)
                    }));
                    display = OutputViewModel.ToBitmapSource(small);
                }
                else
                {
                    display = OutputViewModel.ToBitmapSource(img);
                }
            }
            catch (Exception ex)
            {
                error = ex;
            }
            finally
            {
                if (error is not null) img?.Dispose();
            }

            if (cts.IsCancellationRequested)
            {
                img?.Dispose();
                return;
            }

            Application.Current?.Dispatcher.Invoke(() =>
            {
                _current?.Dispose();
                if (error is not null)
                {
                    _current = null;
                    HasImage = false;
                    Status = $"Couldn't load {Path.GetFileName(path)}: {error.Message}";
                    return;
                }

                _current = img;
                _nativeW = img!.Width;
                _nativeH = img.Height;
                ImageSource = display;
                ImageName = Path.GetFileName(path);
                ImageSize = $"{_nativeW} × {_nativeH} px";
                HasImage = true;
                Status = $"Loaded {_nativeW}×{_nativeH}. Move the mouse to aim the crop box, click to save.";

                // Start the box centred; Relayout/MoveBox re-place it once geometry is known.
                _boxSide = NativeSide();
                _boxNx = (_nativeW - _boxSide) / 2;
                _boxNy = (_nativeH - _boxSide) / 2;
                if (_scale > 0) PlaceBox();
            });
        });
    }

    // ---------------------------------------------------------------- mouse-driven box

    private static int NativeSideOf(int target, int w, int h) => Math.Min(target, Math.Min(w, h));

    private int NativeSide() => NativeSideOf(CropSize, _nativeW, _nativeH);

    /// <summary>The in-image portion of the box (box ∩ image — the box may hang one or both axes off the edge).</summary>
    private Rectangle ClippedRegion()
    {
        int x0 = Math.Max(0, _boxNx);
        int y0 = Math.Max(0, _boxNy);
        int x1 = Math.Min(_nativeW, _boxNx + _boxSide);
        int y1 = Math.Min(_nativeH, _boxNy + _boxSide);
        return new Rectangle(x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
    }

    /// <summary>Recomputes the display mapping for the current container size and re-places the box.</summary>
    /// <param name="containerWidth">Host container width (DIP).</param>
    /// <param name="containerHeight">Host container height (DIP).</param>
    public void Relayout(double containerWidth, double containerHeight)
    {
        if (_current is null) return;
        ComputeGeometry(containerWidth, containerHeight);
        PlaceBox();
    }

    /// <summary>
    /// Centers the crop box on the cursor (the mouse stays over the region being
    /// cropped). The box may straddle the image edge — the crop is clipped to the
    /// in-image portion on save; only at least a 1 px overlap is enforced so a
    /// saved crop can never be empty.
    /// </summary>
    public void MoveBox(double containerWidth, double containerHeight, double cursorX, double cursorY)
    {
        if (_current is null || containerWidth <= 0 || containerHeight <= 0) return;
        ComputeGeometry(containerWidth, containerHeight);

        _boxSide = NativeSide();
        int cx = (int)Math.Round((cursorX - _offX) / _scale);
        int cy = (int)Math.Round((cursorY - _offY) / _scale);
        _boxNx = Math.Clamp(cx - _boxSide / 2, 1 - _boxSide, _nativeW - 1);
        _boxNy = Math.Clamp(cy - _boxSide / 2, 1 - _boxSide, _nativeH - 1);
        PlaceBox();
    }

    /// <summary>Adjusts the crop size from the mouse wheel (&plusmn;<see cref="CropSizeWheelStep"/> px, clamped to [<see cref="CropSizeWheelMin"/>, <see cref="CropSizeWheelMax"/>]).</summary>
    public void WheelCropSize(double delta)
    {
        if (_current is null || delta == 0) return;
        int step = CropSizeWheelStep * (delta > 0 ? 1 : -1);
        int next = Math.Clamp(CropSize + step, CropSizeWheelMin, CropSizeWheelMax);
        if (next != CropSize)
            CropSize = next; // OnCropSizeChanged re-centers the box and refreshes the overlay + preview
    }

    /// <summary>Fit-to-pane, capped at native size (exactly PixelPruner's display_image math).</summary>
    private void ComputeGeometry(double cw, double ch)
    {
        if (_nativeW <= 0 || _nativeH <= 0) return;
        double s = Math.Min(cw / _nativeW, ch / _nativeH);
        _scale = Math.Min(s, 1.0);
        _offX = (cw - _nativeW * _scale) / 2;
        _offY = (ch - _nativeH * _scale) / 2;
    }

    /// <summary>Shows the in-image crop area (box ∩ image) in container DIPs and triggers the live preview.</summary>
    private void PlaceBox()
    {
        if (_scale <= 0) return;
        var r = ClippedRegion();
        if (r.Width <= 0 || r.Height <= 0)
        {
            BoxVisible = false;
            return;
        }
        BoxLeft = _offX + r.X * _scale;
        BoxTop = _offY + r.Y * _scale;
        BoxWidth = r.Width * _scale;
        BoxHeight = r.Height * _scale;
        BoxVisible = true;
        MaybeUpdatePreview(r);
    }

    /// <summary>Aspect-preserving size scaled into (at most) &lt;paramref name="maxSide"&gt;</summary>
    private static (int W, int H) FitInto(int w, int h, int maxSide)
    {
        double sc = (double)maxSide / Math.Max(w, h);
        return (Math.Max(1, (int)Math.Round(w * sc)), Math.Max(1, (int)Math.Round(h * sc)));
    }

    /// <summary>Regenerates the live preview when the boxed native region changed (PixelPruner's update_preview).</summary>
    private void MaybeUpdatePreview(Rectangle region)
    {
        if (_current is null) return;
        if (region.Width <= 0 || region.Height <= 0) return;
        if (region == _lastPreview) return;
        _lastPreview = region;

        _previewCts?.Cancel();
        var cts = new CancellationTokenSource();
        _previewCts = cts;
        var img = _current;
        int target = Math.Min(CropSize, MaxPreviewSide);

        Task.Run(() =>
        {
            ImageSource? src = null;
            try
            {
                using var crop = img.Clone();
                crop.Mutate(ctx => ctx.Crop(region));
                // Aspect-preserving (a box straddling the edge yields a non-square region).
                (int pw, int ph) = FitInto(region.Width, region.Height, target);
                crop.Mutate(ctx => ctx.Resize(pw, ph));
                src = OutputViewModel.ToBitmapSource(crop);
            }
            catch
            {
                // Preview is cosmetic — never fail the tab over it.
                return;
            }

            if (cts.IsCancellationRequested) return;
            Application.Current?.Dispatcher.Invoke(() =>
            {
                if (_lastPreview == region)
                    CropPreview = src;
            });
        });
    }

    // ---------------------------------------------------------------- saving crops

    /// <summary>
    /// Crops the in-image portion of the box (clipped to the image edge when the
    /// box hangs off it) at full resolution, scales it aspect-preserving into the
    /// target size, and saves a PNG to &lt;OutputFolder&gt;/manual_crops/ (PixelPruner's crop_image).
    /// </summary>
    [RelayCommand]
    private void CommitCrop()
    {
        if (_current is null)
        {
            Status = "No image loaded.";
            return;
        }

        try
        {
            // Snapshot the selected pixels immediately. This lets another click queue
            // a different crop while an earlier crop is being resized and written.
            var src = _current;
            var region = ClippedRegion();
            if (region.Width <= 0 || region.Height <= 0)
            {
                Status = "Crop box isn't over the image — move it to save.";
                return;
            }
            int target = CropSize;
            // Aspect-preserving scale into the target size: a box straddling the edge
            // yields a non-square region; a fully-inside box still produces target×target.
            (int outW, int outH) = FitInto(region.Width, region.Height, target);
            string stem = Path.GetFileNameWithoutExtension(_images[_currentIndex]);
            string root = _outputFolder();
            string folder = Path.Combine(string.IsNullOrWhiteSpace(root) ? "output" : root, CropsFolderName);

            Directory.CreateDirectory(folder);

            Image<Rgba32> cut = src.Clone(ctx => ctx.Crop(region));
            _pendingCrops.Enqueue(new PendingCrop
            {
                Image = cut,
                OutputWidth = outW,
                OutputHeight = outH,
                Stem = stem,
                Folder = folder
            });

            Status = _pendingCrops.Count == 1
                ? "Saving crop..."
                : $"Crop queued ({_pendingCrops.Count} pending)...";
            if (!_processingCropQueue)
            {
                _processingCropQueue = true;
                _ = ProcessCropQueueAsync();
            }
        }
        catch (Exception ex)
        {
            Status = $"Couldn't queue crop: {ex.GetType().Name}: {ex.Message}";
        }
    }

    private async Task ProcessCropQueueAsync()
    {
        while (_pendingCrops.Count > 0)
        {
            PendingCrop pending = _pendingCrops.Dequeue();
            Image<Rgba32>? sized = null;
            ImageSource? thumb = null;
            try
            {
                // PixelPruner-style name: cropped_<counter>_<original stem>.png —
                // bump past any leftover from a previous session.
                int n = _cropCounter + 1;
                string path;
                do
                {
                    path = Path.Combine(pending.Folder, $"cropped_{n}_{pending.Stem}.png");
                    n++;
                }
                while (File.Exists(path));

                (sized, thumb) = await Task.Run(() =>
                {
                    var cut = pending.Image;
                    ImageSource? th = null;
                    try
                    {
                        try
                        {
                            using var thumbSrc = cut.Clone();
                            (int tw, int th_s) = FitInto(cut.Width, cut.Height, ThumbnailSize);
                            thumbSrc.Mutate(ctx => ctx.Resize(tw, th_s));
                            th = OutputViewModel.ToBitmapSource(thumbSrc);
                        }
                        catch { /* thumbnail is cosmetic */ }

                        cut.Mutate(ctx => ctx.Resize(pending.OutputWidth, pending.OutputHeight));
                        return (cut, th); // ownership transfers to sized
                    }
                    catch
                    {
                        cut.Dispose();
                        throw;
                    }
                });

                try
                {
                    await Task.Run(() => sized.SaveAsPng(path));
                }
                finally
                {
                    sized.Dispose();
                    sized = null;
                }
                _cropCounter = n - 1;

                Crops.Insert(0, new CropItemViewModel(path, thumb, DeleteCrop));
                HasCrops = true;
                Status = _pendingCrops.Count == 0
                    ? $"Cropped → {path}"
                    : $"Cropped → {path} ({_pendingCrops.Count} pending)";
            }
            catch (ObjectDisposedException)
            {
                pending.Image.Dispose();
                Status = "The image changed while the crop was running — click to save again.";
            }
            catch (Exception ex)
            {
                sized?.Dispose();
                pending.Image.Dispose();
                Status = $"Couldn't save crop: {ex.GetType().Name}: {ex.Message}";
            }
        }

        _processingCropQueue = false;
    }

    /// <summary>Deletes one saved crop (file + list entry) — PixelPruner's delete_crop.</summary>
    private void DeleteCrop(CropItemViewModel item)
    {
        bool wasRemoved = Crops.Remove(item);
        if (!wasRemoved) return;
        if (Crops.Count == 0) HasCrops = false;
        try
        {
            if (File.Exists(item.FilePath)) File.Delete(item.FilePath);
            Status = $"Deleted {Path.GetFileName(item.FilePath)}";
        }
        catch (Exception ex)
        {
            Status = $"Couldn't delete {Path.GetFileName(item.FilePath)}: {ex.Message}";
        }
    }

    // ---------------------------------------------------------------- navigation

    public void Move(int delta)
    {
        if (_imageCount == 0) return;
        // Property (not field) write so PropertyChanged fires for the UI bindings;
        // OnCurrentIndexChanged raises PositionText.
        CurrentIndex = ((CurrentIndex + delta) % ImageCount + ImageCount) % ImageCount;
        LoadCurrent();
    }

    [RelayCommand]
    private void Prev() => Move(-1);

    [RelayCommand]
    private void Next() => Move(1);

    // ---------------------------------------------------------------- actions

    /// <summary>Adds the saved crops to the Process queue so they can be run as source images.</summary>
    [RelayCommand]
    private void AddCropsToQueue()
    {
        if (Crops.Count == 0)
        {
            Status = "No crops saved yet — click the image to make some first.";
            return;
        }
        _queueCrops(Crops.Select(c => c.FilePath));
        Status = $"{Crops.Count} crop(s) added to the Process queue — hit Start there.";
    }

    [RelayCommand]
    private void OpenCropsFolder()
    {
        string root = _outputFolder();
        string folder = Path.Combine(string.IsNullOrWhiteSpace(root) ? "output" : root, CropsFolderName);
        try
        {
            Directory.CreateDirectory(folder);
            System.Diagnostics.Process.Start("explorer.exe", $"\"{folder}\"");
        }
        catch (Exception ex)
        {
            Status = $"Couldn't open crops folder: {ex.Message}";
        }
    }

    /// <summary>Opens the file dialog; chosen files are added to the Process queue (which feeds this tab).</summary>
    [RelayCommand]
    private void OpenImage()
    {
        var dlg = new OpenFileDialog
        {
            Multiselect = true,
            Title = "Select images to crop",
            Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tiff|All files|*.*"
        };
        if (dlg.ShowDialog() == true)
            _queueCrops(dlg.FileNames);
    }
}
