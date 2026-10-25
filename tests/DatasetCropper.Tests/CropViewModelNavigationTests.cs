using System.IO;
using System.Linq;
using DatasetCropper.ViewModels;
using Xunit;

namespace DatasetCropper.Tests;

/// <summary>
/// Regression: with multiple queued images the Crop tab must expose navigation —
/// <c>CanNavigate</c> / <c>PositionText</c> reflect the whole list and Prev/Next wrap.
/// </summary>
public class CropViewModelNavigationTests : IDisposable
{
    private readonly string _dir;

    public CropViewModelNavigationTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "DatasetCropper_crop_nav_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        foreach (var name in new[] { "a.png", "b.png" })
            File.WriteAllBytes(Path.Combine(_dir, name), new byte[8]); // bytes don't matter: Refresh() only checks existence
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, true); } catch { }
    }

    private CropViewModel NewVm()
    {
        var paths = new[] { Path.Combine(_dir, "a.png"), Path.Combine(_dir, "b.png") };
        var vm = new CropViewModel(() => paths.ToList(), () => _dir, _ => { });
        vm.Refresh();
        return vm;
    }

    [Fact]
    public void Refresh_TwoImages_CanNavigateAndPositionSet()
    {
        var vm = NewVm();

        Assert.True(vm.CanNavigate);
        Assert.Equal("1 of 2", vm.PositionText);
    }

    [Fact]
    public void Move_ForwardsAndBack_Wraps()
    {
        var vm = NewVm();

        vm.Move(1);
        Assert.Equal("2 of 2", vm.PositionText);

        vm.Move(1); // wrap to first
        Assert.Equal("1 of 2", vm.PositionText);

        vm.Move(-1); // wrap to last
        Assert.Equal("2 of 2", vm.PositionText);
    }

    [Fact]
    public void Refresh_TellsTheBindingCanNavigate()
    {
        // The Prev/Next buttons bind IsEnabled to CanNavigate, which is only
        // re-evaluated when PropertyChanged fires. Refresh() must raise it
        // (regression: a bare field write left the buttons permanently disabled).
        var paths = new[] { Path.Combine(_dir, "a.png"), Path.Combine(_dir, "b.png" ) };
        var vm = new CropViewModel(() => paths.ToList(), () => _dir, _ => { });
        var fired = new List<string>();
        vm.PropertyChanged += (_, e) => fired.Add(e.PropertyName!);
        vm.Refresh();

        Assert.Contains(nameof(CropViewModel.ImageCount), fired);
        Assert.Contains(nameof(CropViewModel.CanNavigate), fired);
        Assert.Contains(nameof(CropViewModel.PositionText), fired);
    }

    [Fact]
    public void PropertyChanged_FiresForNavigationMembers()
    {
        var vm = NewVm();
        var fired = new List<string>();
        vm.PropertyChanged += (_, e) => fired.Add(e.PropertyName!);

        vm.Move(1);

        Assert.Contains(nameof(CropViewModel.PositionText), fired);
        Assert.Contains(nameof(CropViewModel.CurrentIndex), fired);
    }
}
