using System;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace DatasetCropper.ViewModels;

/// <summary>One saved crop in the Crop tab's thumbnail list (like one of PixelPruner's crop tiles).</summary>
public partial class CropItemViewModel : ObservableObject
{
    /// <summary>Callback into <see cref="CropViewModel"/> to remove this item and delete its file.</summary>
    private readonly Action<CropItemViewModel> _onRemove;

    public string FilePath { get; }

    /// <summary>Path file name, shown under the thumbnail.</summary>
    public string FileName => System.IO.Path.GetFileName(FilePath);

    /// <summary>~256px preview, produced when the crop is saved (background thread).</summary>
    [ObservableProperty]
    private ImageSource? _thumbnail;

    public CropItemViewModel(string filePath, ImageSource? thumbnail, Action<CropItemViewModel> onRemove)
    {
        FilePath = filePath;
        Thumbnail = thumbnail;
        _onRemove = onRemove;
    }

    [RelayCommand]
    private void Delete() => _onRemove(this);
}
