using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace DatasetCropper.Converters;

/// <summary>True → Collapsed, false → Visible (the standard BooleanToVisibilityConverter ignores ConverterParameter, so this inverts explicitly).</summary>
public sealed class InverseBoolToVisibilityConverter : IValueConverter
{
    public static readonly InverseBoolToVisibilityConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is bool b && b
            ? Visibility.Collapsed
            : Visibility.Visible;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is Visibility v && v == Visibility.Collapsed;
}
