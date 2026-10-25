using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace DatasetCropper.Converters;

/// <summary>
/// Returns <see cref="Visibility.Visible"/> when the bound value is a
/// non-null, non-whitespace string; otherwise <see cref="Visibility.Collapsed"/>.
/// Used to hide the "Open" button on a queue row until an output folder is
/// known.
/// </summary>
public sealed class NullStringToCollapsedConverter : IValueConverter
{
    public static readonly NullStringToCollapsedConverter Instance = new();

    private NullStringToCollapsedConverter() { }

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        (value is string s && !string.IsNullOrWhiteSpace(s))
            ? Visibility.Visible
            : Visibility.Collapsed;

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
