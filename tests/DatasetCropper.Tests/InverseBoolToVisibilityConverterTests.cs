using System.Windows;
using DatasetCropper.Converters;
using Xunit;

namespace DatasetCropper.Tests;

public class InverseBoolToVisibilityConverterTests
{
    private static readonly InverseBoolToVisibilityConverter C = InverseBoolToVisibilityConverter.Instance;

    [Fact]
    public void True_IsCollapsed_False_IsVisible()
    {
        Assert.Equal(Visibility.Collapsed, C.Convert(true, typeof(Visibility), null, null));
        Assert.Equal(Visibility.Visible, C.Convert(false, typeof(Visibility), null, null));
    }

    [Fact]
    public void ConvertBack_Inverts()
    {
        // ConvertBack is the inverse of Convert: Collapsed came from true, Visible from false.
        Assert.Equal(true, C.ConvertBack(Visibility.Collapsed, typeof(bool), null, null));
        Assert.Equal(false, C.ConvertBack(Visibility.Visible, typeof(bool), null, null));
    }
}
