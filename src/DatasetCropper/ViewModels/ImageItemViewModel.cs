using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DatasetCropper.Models;

namespace DatasetCropper.ViewModels;

/// <summary>One row in the image queue.</summary>
public partial class ImageItemViewModel : ObservableObject
{
    public string FilePath { get; }
    public string FileName => System.IO.Path.GetFileName(FilePath);
    public int Index { get; }

    [ObservableProperty]
    private ImageStatus _status = ImageStatus.Queued;

    [ObservableProperty]
    private string _detail = "Queued";

    /// <summary>The output directory where the processed images are saved.</summary>
    [ObservableProperty]
    private string? _outputFolder;

    [RelayCommand]
    private void OpenFolder()
    {
        if (!string.IsNullOrWhiteSpace(OutputFolder) && System.IO.Directory.Exists(OutputFolder))
        {
            System.Diagnostics.Process.Start("explorer.exe", $"\"{OutputFolder}\"");
        }
    }

    public ImageItemViewModel(int index, string filePath)
    {
        Index = index;
        FilePath = filePath;
    }

    public void SetStatus(ImageStatus status, string? detail = null)
    {
        Status = status;
        Detail = detail ?? status.ToString();
    }
}
