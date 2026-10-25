using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using DatasetCropper.Imaging;
using DatasetCropper.Ml;
using DatasetCropper.Models;
using DatasetCropper.Services;

namespace DatasetCropper.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly AppSettings _settings;
    private MlHub? _ml;
    private ProcessingService? _service;

    public ObservableCollection<ImageItemViewModel> Items { get; } = new();

    // --- Settings (bound) ---
    [ObservableProperty] private string _conceptsText = "";
    [ObservableProperty] private string _subjectConceptsText = "";
    [ObservableProperty] private string _maskTermsText = "";
    [ObservableProperty] private string _outputFolder = "";
    [ObservableProperty] private bool _appendCaptions;
    [ObservableProperty] private bool _overwriteOutputs;
    [ObservableProperty] private string _modelsFolder = "";
    [ObservableProperty] private bool _useTopaz;
    [ObservableProperty] private string _topazPath = "";
    [ObservableProperty] private string _topazArgs = "--upscale";
    [ObservableProperty] private bool _useGpu;
    [ObservableProperty] private double _wd14Threshold;
    [ObservableProperty] private double _dinoBoxThreshold;
    [ObservableProperty] private double _dinoTextThreshold;
    [ObservableProperty] private int _minTileOverlap;
    [ObservableProperty] private int _minimumSubjectResolution;
    [ObservableProperty] private string _captionPrefix = "";

    // --- Processing stages (checkboxes above the log; all default on) ---
    [ObservableProperty] private bool _stageTopaz = true;
    [ObservableProperty] private bool _stageDetect = true;
    [ObservableProperty] private bool _stageSubjects = true;
    [ObservableProperty] private bool _stageMasks = true;
    [ObservableProperty] private bool _stageCaptions = true;
    [ObservableProperty] private bool _stageDedupe;
    // UI-only checkbox: emit an extra concept-centered crop when multiple concepts match.
    [ObservableProperty] private bool _stageMultiCrop = false;

    // --- Mask model selection (which model generates the grayscale concept masks) ---
    [ObservableProperty] private MaskModel _selectedMaskModel = MaskModel.Sam;
    public MaskModel[] MaskModelOptions { get; } = Enum.GetValues<MaskModel>().ToArray();
    [ObservableProperty] private string _maskModelDescription = "";

    [ObservableProperty] private CaptionModel _selectedCaptionModel = CaptionModel.Wd14;
    public CaptionModel[] CaptionModelOptions { get; } = Enum.GetValues<CaptionModel>().ToArray();
    [ObservableProperty] private string _captionModelDescription = "";

    partial void OnSelectedMaskModelChanged(MaskModel value)
        => MaskModelDescription = DescribeMaskModel(value);

    partial void OnSelectedCaptionModelChanged(CaptionModel value)
        => CaptionModelDescription = value switch
        {
            CaptionModel.VitGpt2 => "Apache-2.0 ViT-GPT2 image captioner. Generates a short natural-language sentence using a quantized ONNX vision encoder and GPT-2 decoder.",
            _ => "WD14 writes confidence-filtered booru tags rather than a natural-language sentence.",
        };

    /// <summary>Size + advantages of each mask model, for the Settings tab description.</summary>
    internal static string DescribeMaskModel(MaskModel model) => model switch
    {
        MaskModel.ClipSeg =>
            "≈547 MB (3 files). Semantic text segmentation: the mask follows the *meaning* of the concept and covers the whole object, including occluded parts. Works for every concept, but may include all instances of that concept in the frame.",
        MaskModel.RembgHuman =>
            "≈176 MB (1 file) — the smallest and fastest. Human specialist: excellent for people, but only meaningful when the concept is a human — for other concepts it will mask every person in the frame.",
        _ =>
            "≈360 MB (2 files). Instance segmentation from box/point prompts: masks the object DINO found, wherever it is. The sharpest edges and it works for every concept — the best general-purpose default.",
    };

    // --- Detector selection (which model finds the concept/subject boxes) ---
    [ObservableProperty] private DetectorModel _selectedDetectorModel = DetectorModel.GroundingDino;
    public DetectorModel[] DetectorModelOptions { get; } = Enum.GetValues<DetectorModel>().ToArray();
    [ObservableProperty] private string _detectorModelDescription = "";

    partial void OnSelectedDetectorModelChanged(DetectorModel value)
        => DetectorModelDescription = DescribeDetectorModel(value);

    internal static string DescribeDetectorModel(DetectorModel model) => model switch
    {
        DetectorModel.Yolo =>
            "≈6 MB, fast — one inference pass (no per-concept runs). Closed vocabulary: the 80 COCO classes, so a concept must resemble one of them (person, car, cat, dog, …). Excellent for people/vehicles/animals; won't find open concepts like \"sailboat\".",
        _ =>
            "≈1.1 GB, open-vocabulary: detects any concept you type (person, sailboat, koi, …). Slower (one run per concept) and the largest model — but the best general-purpose default.",
    };

    // --- Model availability (populated by CheckModels) ---
    [ObservableProperty] private string _modelsStatus = "";

    /// <summary>Backs the Output tab (thumbnail grid + preview + Del-delete).</summary>
    public OutputViewModel Output { get; }

    /// <summary>Backs the Crop tab (manual 1:1 crops saved for processing as source images).</summary>
    public CropViewModel Crop { get; }

    // --- Runtime state ---
    [ObservableProperty] private bool _isRunning;
    [ObservableProperty] private string _statusText = "Idle";
    [ObservableProperty] private double _overallProgress;
    [ObservableProperty] private string _logText = "";

    public MainViewModel()
    {
        _settings = SettingsStore.Load();
        Output = new OutputViewModel(() => OutputFolder, LiveConcepts);
        Crop = new CropViewModel(
            () => Items.Select(i => i.FilePath).ToList(),
            () => OutputFolder,
            files => AddFiles(files));
        ApplySettingsToUi();
        MaskModelDescription = DescribeMaskModel(SelectedMaskModel);
        CaptionModelDescription = DescribeCaptionModel(SelectedCaptionModel);
        DetectorModelDescription = DescribeDetectorModel(SelectedDetectorModel);
    }

    internal static string DescribeCaptionModel(CaptionModel model) => model switch
    {
        CaptionModel.VitGpt2 => "Apache-2.0 ViT-GPT2 image captioner. Generates a short natural-language sentence using a quantized ONNX vision encoder and GPT-2 decoder.",
        _ => "WD14 writes confidence-filtered booru tags rather than a natural-language sentence.",
    };

    /// <summary>The live concept list behind the Output tab's 1-9 assignment (Composition, then Subjects).</summary>
    private List<string> LiveConcepts()
    {
        var list = new List<string>();
        foreach (var text in new[] { ConceptsText, SubjectConceptsText })
            list.AddRange(text
                .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim())
                .Where(t => t.Length > 0));
        return list;
    }

    private void ApplySettingsToUi()
    {
        ConceptsText = string.Join("\n", _settings.Concepts);
        SubjectConceptsText = string.Join("\n", _settings.SubjectConcepts);
        MaskTermsText = string.Join("\n", _settings.MaskTerms);
        OutputFolder = _settings.OutputFolder;
        AppendCaptions = _settings.AppendCaptions;
        OverwriteOutputs = _settings.OverwriteOutputs;
        ModelsFolder = _settings.ModelsFolder;
        UseTopaz = _settings.UseTopaz;
        TopazPath = _settings.TopazPath;
        TopazArgs = _settings.TopazArgs;
        UseGpu = _settings.UseGpu;
        Wd14Threshold = _settings.Wd14Threshold;
        DinoBoxThreshold = _settings.DinoBoxThreshold;
        DinoTextThreshold = _settings.DinoTextThreshold;
        MinTileOverlap = _settings.MinTileOverlap;
        MinimumSubjectResolution = _settings.MinimumSubjectResolution;
        CaptionPrefix = _settings.CaptionPrefix;
        StageTopaz = _settings.StageTopaz;
        StageDetect = _settings.StageDetect;
        StageSubjects = _settings.StageSubjects;
        StageMasks = _settings.StageMasks;
        StageCaptions = _settings.StageCaptions;
        StageDedupe = _settings.StageDedupe;
        StageMultiCrop = _settings.StageMultiCrop;
        SelectedMaskModel = _settings.MaskModel;
        SelectedCaptionModel = _settings.CaptionModel;
        SelectedDetectorModel = _settings.DetectorModel;
    }

    private void SyncSettingsFromUi()
    {
        _settings.Concepts = ConceptsText
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(c => c.Trim())
            .Where(c => c.Length > 0)
            .ToList();
        _settings.MaskTerms = MaskTermsText
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(c => c.Trim())
            .Where(c => c.Length > 0)
            .ToList();
        _settings.SubjectConcepts = SubjectConceptsText
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(c => c.Trim())
            .Where(c => c.Length > 0)
            .ToList();
        _settings.OutputFolder = OutputFolder;
        _settings.AppendCaptions = AppendCaptions;
        _settings.OverwriteOutputs = OverwriteOutputs;
        _settings.ModelsFolder = ModelsFolder;
        _settings.UseTopaz = UseTopaz;
        _settings.TopazPath = TopazPath;
        _settings.TopazArgs = TopazArgs;
        _settings.UseGpu = UseGpu;
        _settings.Wd14Threshold = Wd14Threshold;
        _settings.DinoBoxThreshold = DinoBoxThreshold;
        _settings.DinoTextThreshold = DinoTextThreshold;
        _settings.MinTileOverlap = MinTileOverlap;
        _settings.MinimumSubjectResolution = Math.Max(1, MinimumSubjectResolution);
        _settings.CaptionPrefix = (CaptionPrefix ?? "").Trim();
        _settings.StageTopaz = StageTopaz;
        _settings.StageDetect = StageDetect;
        _settings.StageSubjects = StageSubjects;
        _settings.StageMasks = StageMasks;
        _settings.StageCaptions = StageCaptions;
        _settings.StageDedupe = StageDedupe;
        _settings.MaskModel = SelectedMaskModel;
        _settings.CaptionModel = SelectedCaptionModel;
        _settings.DetectorModel = SelectedDetectorModel;
        _settings.StageMultiCrop = StageMultiCrop;
        SettingsStore.Save(_settings);
    }

    [RelayCommand]
    private void AddImages()
    {
        var dlg = new OpenFileDialog { Multiselect = true, Title = "Select images",
            Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tiff|All files|*.*" };
        if (dlg.ShowDialog() == true)
            AddFiles(dlg.FileNames);
    }

    public void AddFiles(IEnumerable<string> files)
    {
        // Dropped folders expand recursively to their files (CLI behaves the same).
        foreach (var f in ImageFileFilter.FilterFiles(ImageFileFilter.ExpandDirectories(files)))
        {
            if (!File.Exists(f)) continue;
            if (Items.Any(i => i.FilePath == f)) continue;
            Items.Add(new ImageItemViewModel(Items.Count, f));
        }
        StatusText = $"{Items.Count} image(s) queued.";
        Crop.OnQueueChanged();
    }

    [RelayCommand]
    private void ClearQueue()
    {
        if (IsRunning) return;
        Items.Clear();
        StatusText = "Idle";
        Crop.OnQueueChanged();
    }

    /// <summary>Selection held in sync with the queue ListView (extended selection).</summary>
    public ObservableCollection<ImageItemViewModel> QueueSelection { get; } = new();

    public void SyncQueueSelection(IEnumerable<ImageItemViewModel> selected)
    {
        var list = selected.ToList();
        if (list.Count == QueueSelection.Count
            && list.All(i => QueueSelection.Contains(i))) return;
        QueueSelection.Clear();
        foreach (var i in list) QueueSelection.Add(i);
    }

    /// <summary>Removes the currently selected queue items (Delete key). Blocked while a run is active.</summary>
    [RelayCommand]
    private void RemoveSelected()
    {
        if (IsRunning)
        {
            StatusText = "Wait for the run to finish before removing queue items.";
            return;
        }
        if (QueueSelection.Count == 0)
        {
            StatusText = "Select one or more images in the queue first (Delete removes them).";
            return;
        }
        var removed = QueueSelection.ToList();
        foreach (var item in removed) Items.Remove(item);
        QueueSelection.Clear();
        StatusText = $"{removed.Count} image(s) removed; {Items.Count} left in the queue.";
        Crop.OnQueueChanged();
    }

    [RelayCommand]
    private void ChooseOutputFolder()
    {
        var dlg = new OpenFolderDialog { Title = "Choose output folder" };
        if (!string.IsNullOrEmpty(OutputFolder) && Directory.Exists(OutputFolder))
            dlg.InitialDirectory = OutputFolder;
        if (dlg.ShowDialog() == true)
            OutputFolder = dlg.FolderName;
    }

    [RelayCommand]
    private void OpenOutputFolder()
    {
        if (string.IsNullOrWhiteSpace(OutputFolder))
        {
            StatusText = "No output folder set.";
            return;
        }
        try
        {
            Directory.CreateDirectory(OutputFolder);
            System.Diagnostics.Process.Start("explorer.exe", $"\"{OutputFolder}\"");
        }
        catch (Exception ex)
        {
            StatusText = $"Couldn't open output folder: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ChooseModelsFolder()
    {
        var dlg = new OpenFolderDialog { Title = "Choose models folder" };
        if (!string.IsNullOrEmpty(ModelsFolder) && Directory.Exists(ModelsFolder))
            dlg.InitialDirectory = ModelsFolder;
        if (dlg.ShowDialog() == true)
            ModelsFolder = dlg.FolderName;
    }

    [RelayCommand]
    private void CheckModels()
    {
        ModelsStatus = string.Empty;

        if (string.IsNullOrWhiteSpace(ModelsFolder) || !Directory.Exists(ModelsFolder))
        {
            ModelsStatus = "Models folder not set.";
            AppendLog("Model Check.. Folder not set.");
            return;
        }
        var results = ModelChecker.Check(ModelsFolder);
        foreach(var result in results)
        {
            ModelsStatus = string.Join("   ", results.Select(r => $"{r.Name}: {r.Status}"));
            if (result.Status != "ready")
            {
                AppendLog($"  Model {result.Name} missing or incomplete - {result.Status}");
            }
        }
    }

    /// <summary>
    /// Core download path: prompts the user to confirm, then downloads every
    /// missing model file into the models folder. Assumes &lt;IsRunning&gt;
    /// is already true and the log area is ready. Refreshes &lt;ModelsStatus&gt;.
    /// </summary>
    private async Task<bool> DownloadMissingAsync()
    {
        if (string.IsNullOrWhiteSpace(ModelsFolder))
        {
            AppendLog("  Download skipped: models folder not set.");
            return false;
        }
        Directory.CreateDirectory(ModelsFolder);
        var missing = ModelDownloader.FindMissing(ModelsFolder);
        if (missing.Count == 0)
        {
            AppendLog("  All models already present on disk.");
            return true;
        }
        var names = missing.Select(m => m.Name).Distinct().ToList();
        var res = MessageBox.Show(
            $"Missing models for: {string.Join(", ", names)}\n\n" +
            "Download now?\n(Several gigabytes — progress will appear in the log.)",
            "Download missing models?", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (res != MessageBoxResult.Yes)
        {
            AppendLog("  Download skipped by user.");
            return false;
        }
        using var cts = new CancellationTokenSource();

        // Replaces the last line of the log with the current progress value
        // (in place — not appended). This works whether the last line is the
        // "Downloading ..." header (first call) or a previous bar update
        // (subsequent calls), and always ends with a newline so the next
        // AppendLog call lands on a fresh line.
        void UpdateProgress(string line)
        {
            string text = LogText;
            if (text.Length == 0)
            {
                LogText = line + Environment.NewLine;
                return;
            }
            int stripped = text.EndsWith(Environment.NewLine, StringComparison.Ordinal)
                ? text.Length - Environment.NewLine.Length
                : text.Length;
            int cut = stripped > 0 ? text[..stripped].LastIndexOf('\n') + 1 : 0;
            LogText = text[..cut] + line + Environment.NewLine;
        }

        try
        {
            int n = await Task.Run(() =>
                ModelDownloader.Download(ModelsFolder, AppendLog, UpdateProgress, cts.Token), cts.Token);
            AppendLog($"  Download complete: {n} file(s) written.");
            return true;
        }
        catch (OperationCanceledException)
        {
            AppendLog("  Download cancelled.");
            return false;
        }
        catch (Exception ex)
        {
            AppendLog($"  Download failed: {ex.Message}");
            return false;
        }
        finally
        {
            CheckModels();
        }
    }

    /// <summary>
    /// Manual "download missing models" UI action.
    /// </summary>
    [RelayCommand]
    private async Task DownloadMissingModels()
    {
        if (IsRunning) return;
        if (string.IsNullOrWhiteSpace(ModelsFolder))
        {
            StatusText = "Set a models folder first (Choose models folder).";
            return;
        }
        IsRunning = true;
        BeginLogRun();
        try
        {
            await DownloadMissingAsync();
            StatusText = "Model download finished.";
        }
        finally
        {
            IsRunning = false;
        }
    }

    // Re-check availability whenever the models folder changes (typed or browsed).
    partial void OnModelsFolderChanged(string value) => CheckModels();

    // Keep the Output tab in sync when the output folder is retargeted.
    partial void OnOutputFolderChanged(string value) => Output.RefreshAll();

    [RelayCommand]
    private void ChooseTopazPath()
    {
        var dlg = new OpenFileDialog { Title = "Locate tpai.exe", Filter = "tpai.exe|tpai.exe|All files|*.*" };
        if (dlg.ShowDialog() == true)
            TopazPath = dlg.FileName;
    }

    [RelayCommand]
    private async Task Start()
    {
        if (IsRunning) return;
        if (Items.Count == 0) { StatusText = "Nothing to process — add some images first."; return; }
        SyncSettingsFromUi();
        if (_settings.Concepts.Count == 0) { StatusText = "Define at least one concept."; return; }

        IsRunning = true;
        BeginLogRun();
        LogText = "";
        AppendLog("Loading models...");
        foreach (var item in Items) item.SetStatus(ImageStatus.Queued);

        // If any models are missing, offer to download them now.
        // DownloadMissingAsync returns false when the user declined or the
        // download failed — that's fine, we continue with whatever we have.
        if (ModelDownloader.FindMissing(ModelsFolder).Count > 0)
        {
            await DownloadMissingAsync();
        }

        // Load models on a background thread (can be slow).
        var ml = await Task.Run(() => MlHub.Load(ModelsFolder, _settings.UseGpu, AppendLog, _settings.MaskModel, _settings.DetectorModel, _settings.CaptionModel));
        foreach (var err in ml.LoadErrors) AppendLog("  " + err);
        _ml = ml;
        _service = new ProcessingService(_settings, ml);
        _service.StageMultiCrop = StageMultiCrop;

        var progress = new Progress<ProgressReport>(r =>
        {
            if (r.CurrentImage is not null)
            {
                var item = Items.FirstOrDefault(i => i.FilePath == r.CurrentImage);
                if (item is not null)
                {
                    var newStatus = MapStage(r.Stage);
                    item.SetStatus(newStatus, r.Message);

                    // Expose the output folder as soon as the image is done processing
                    if (newStatus == ImageStatus.Done)
                        item.OutputFolder = OutputFolder;
                }
            }
            if (r.Total > 0)
            {
                OverallProgress = r.OverallFraction;
                StatusText = $"{r.Completed}/{r.Total} done";
            }
        });

        AppendLog("Starting pipeline...");
        var files = Items.Select(i => i.FilePath).ToList();
        _service.Run(files, progress, AppendLog);
        await _service.LastTask;

        // Items that never started (e.g. cancelled mid-run) still show "Queued" —
        // give them a clear terminal state and detail.
        int skipped = 0;
        foreach (var item in Items)
        {
            if (item.Status == ImageStatus.Queued)
            {
                item.SetStatus(ImageStatus.Skipped, "Not processed (run stopped before this item)");
                skipped++;
            }
        }

        IsRunning = false;
        OverallProgress = 1;

        int done = Items.Count(i => i.Status == ImageStatus.Done);
        int errors = Items.Count(i => i.Status == ImageStatus.Error);
        StatusText = $"Finished: {done} done, {errors} error{(errors != 1 ? "s" : "")}, {skipped} skipped.";
        Output.RefreshAll();
        ml.Dispose();
        _ml = null;
    }

    [RelayCommand]
    private void Cancel()
    {
        _service?.Cancel();
        StatusText = "Cancelling...";
    }

    private ImageStatus MapStage(string? stage) => stage switch
    {
        "Upscaling" => ImageStatus.Upscaling,
        "Detecting" => ImageStatus.Detecting,
        "Detecting subjects" => ImageStatus.Detecting,
        "Cropping" => ImageStatus.Cropping,
        "Masking" => ImageStatus.Masking,
        "Tagging" => ImageStatus.Tagging,
        "Done" => ImageStatus.Done,
        "Error" => ImageStatus.Error,
        _ => ImageStatus.Queued
    };

    private void AppendLog(string line)
    {
        LogText = LogText.Length > 20000 ? LogText[^8000..] : LogText;
        LogText += line + Environment.NewLine;
        // Persist for post-run debugging (best effort — never break the pipeline).
        try { File.AppendAllText(SettingsStore.LogPath, line + Environment.NewLine); }
        catch { /* ignore */ }
    }

    /// <summary>
    /// Starts a fresh section in the rolling log file, capping its size so it
    /// doesn't grow without bound across many runs.
    /// </summary>
    private static void BeginLogRun()
    {
        try
        {
            Directory.CreateDirectory(SettingsStore.SettingsDir);
            var fi = new FileInfo(SettingsStore.LogPath);
            if (fi.Exists && fi.Length > 2 * 1024 * 1024)
                File.WriteAllText(SettingsStore.LogPath, "");
            File.AppendAllText(SettingsStore.LogPath,
                "\n==== " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " — run started ====" + Environment.NewLine);
        }
        catch { /* best effort */ }
    }
}
