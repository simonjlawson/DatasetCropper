using System.Globalization;
using System.Windows.Data;

namespace DatasetCropper.Converters;

/// <summary>Inverts a boolean (for enabling a button when IsRunning is false).</summary>
public sealed class InverseBoolConverter : IValueConverter
{
    public static readonly InverseBoolConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is bool b ? !b : value;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is bool b ? !b : value;
}
