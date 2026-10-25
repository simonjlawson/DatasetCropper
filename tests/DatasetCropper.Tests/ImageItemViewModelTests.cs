using DatasetCropper.Models;
using DatasetCropper.ViewModels;
using Xunit;

namespace DatasetCropper.Tests;

public class ImageItemViewModelTests
{
    [Fact]
    public void Ctor_StoresIndexAndFilePath()
    {
        var vm = new ImageItemViewModel(3, @"C:\pics\cat.jpg");

        Assert.Equal(3, vm.Index);
        Assert.Equal(@"C:\pics\cat.jpg", vm.FilePath);
    }

    [Fact]
    public void Ctor_FileNameIsLastSegmentOfPath()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        Assert.Equal("cat.jpg", vm.FileName);
    }

    [Fact]
    public void Ctor_DefaultStatusIsQueued()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        Assert.Equal(ImageStatus.Queued, vm.Status);
    }

    [Fact]
    public void Ctor_DefaultDetailIsQueued()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        Assert.Equal("Queued", vm.Detail);
    }

    [Fact]
    public void SetStatus_WithDetail_UsesProvidedDetail()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        vm.SetStatus(ImageStatus.Error, "boom");

        Assert.Equal("boom", vm.Detail);
    }

    [Fact]
    public void SetStatus_WithDetail_UpdatesStatusAndPreservesDetail()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        vm.SetStatus(ImageStatus.Done, "saved 4 crops");

        Assert.Equal(ImageStatus.Done, vm.Status);
    }

    [Fact]
    public void SetStatus_WithoutDetail_DetailsStatusName()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        vm.SetStatus(ImageStatus.Detecting);

        Assert.Equal(ImageStatus.Detecting.ToString(), vm.Detail);
    }

    [Fact]
    public void SetStatus_WithoutDetail_UpdatesStatus()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");

        vm.SetStatus(ImageStatus.Masking);

        Assert.Equal(ImageStatus.Masking, vm.Status);
    }

    [Fact]
    public void SetStatus_RaisesPropertyChangedForStatus()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");
        bool raised = false;
        vm.PropertyChanged += (_, e) => raised |= e.PropertyName == nameof(ImageItemViewModel.Status);

        vm.SetStatus(ImageStatus.Tagging);

        Assert.True(raised);
    }

    [Fact]
    public void SetStatus_RaisesPropertyChangedForDetail()
    {
        var vm = new ImageItemViewModel(0, @"C:\pics\cat.jpg");
        bool raised = false;
        vm.PropertyChanged += (_, e) => raised |= e.PropertyName == nameof(ImageItemViewModel.Detail);

        vm.SetStatus(ImageStatus.Tagging, "4 tags");

        Assert.True(raised);
    }
}
