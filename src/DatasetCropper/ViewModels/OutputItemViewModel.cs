using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace DatasetCropper.ViewModels;

/// <summary>One cell in the Output tab's thumbnail grid.</summary>
public partial class OutputItemViewModel : ObservableObject
{
    public string FilePath { get; }

    /// <summary>Path relative to the output root, e.g. "cat/sample_000.png".</summary>
    public string RelativePath { get; }

    /// <summary>Small (256px) preview; filled in on a background thread after the grid is shown.</summary>
    [ObservableProperty]
    private ImageSource? _thumbnail;

    public OutputItemViewModel(string filePath, string relativePath)
    {
        FilePath = filePath;
        RelativePath = relativePath;
    }
}
