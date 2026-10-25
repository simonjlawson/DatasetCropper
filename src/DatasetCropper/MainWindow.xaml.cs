using System.Linq;
using System.Windows;
using System.Windows.Input;
using DatasetCropper.ViewModels;

namespace DatasetCropper;

public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel;

    public MainWindow()
    {
        InitializeComponent();
        _viewModel = new MainViewModel();
        DataContext = _viewModel;
        _viewModel.Output.SelectItemRequested += (_, item) =>
            OutputList.SelectedItem = item;
    }

    private void Window_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop)
            ? DragDropEffects.Copy
            : DragDropEffects.None;
        e.Handled = true;
    }

    private void Window_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop)!;
            _viewModel.AddFiles(files);
        }
        e.Handled = true;
    }

    // --- Queue list (Process tab)

    private void QueueList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (sender is System.Windows.Controls.ListView list)
            _viewModel.SyncQueueSelection(list.SelectedItems.Cast<ImageItemViewModel>());
    }

    private void QueueList_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (Keyboard.Modifiers != ModifierKeys.None) return;
        if (e.Key == Key.Delete)
        {
            _viewModel.RemoveSelectedCommand.Execute(null);
            e.Handled = true;
        }
    }

    // --- Log (Process tab): follow the latest output until the user scrolls up.
    private bool _logFollow = true;
    private int _lastLogLength;

    private void LogBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (sender is not System.Windows.Controls.TextBox box || box.ActualHeight <= 0) return;
        string text = box.Text;
        // A near-empty log means a run reset it — re-follow the new output.
        if (text.Length <= 200) _logFollow = true;
        _lastLogLength = text.Length;
        if (!_logFollow) return;
        // Background priority: the text update has laid out by then, so
        // ScrollToEnd is meaningful and this doesn't interleave with input.
        box.Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Background,
            new System.Action(box.ScrollToEnd));
    }

    private void LogBox_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        => SetLogFollow(e.Delta < 0 ? false : IsLogAtBottom());

    private void LogBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key is Key.Up or Key.PageUp or Key.Home) SetLogFollow(false);
        else if (e.Key is Key.Down or Key.PageDown or Key.End) SetLogFollow(IsLogAtBottom());
    }

    private void SetLogFollow(bool follow)
    {
        if (follow == _logFollow) return;
        _logFollow = follow;
        if (follow)
            LogBox.Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Background,
                new System.Action(LogBox.ScrollToEnd));
    }

    private bool IsLogAtBottom()
    {
        if (FindVisualChild<System.Windows.Controls.ScrollViewer>(LogBox)
            is { ScrollableHeight: > 0 } sv)
            return sv.VerticalOffset + sv.ViewportHeight >= sv.ExtentHeight - 1;
        return true; // nothing to scroll: content fits or the template isn't up yet
    }

    private static T? FindVisualChild<T>(System.Windows.Media.Visual root) where T : System.Windows.Media.Visual
    {
        for (int i = 0; i < System.Windows.Media.VisualTreeHelper.GetChildrenCount(root); i++)
        {
            var child = System.Windows.Media.VisualTreeHelper.GetChild(root, i);
            if (child is T found) return found;
            if (child is System.Windows.Media.Visual visual)
            {
                var nested = FindVisualChild<T>(visual);
                if (nested is not null) return nested;
            }
        }
        return null;
    }

    private void OutputTab_Loaded(object sender, RoutedEventArgs e) =>
        _viewModel.Output.Refresh();

    // --- Crop tab

    private void CropTab_Loaded(object sender, RoutedEventArgs e) =>
        _viewModel.Crop.Refresh();

    private void CropCanvas_MouseMove(object sender, MouseEventArgs e)
    {
        // Note: the host is a Grid (Panel), not a Control — use FrameworkElement.
        if (sender is System.Windows.FrameworkElement host)
        {
            var pos = e.GetPosition(host);
            _viewModel.Crop.MoveBox(host.ActualWidth, host.ActualHeight, pos.X, pos.Y);
        }
    }

    private void CropCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e) =>
        _viewModel.Crop.CommitCropCommand.Execute(null);

    private void CropCanvas_MouseWheel(object sender, MouseWheelEventArgs e) =>
        _viewModel.Crop.WheelCropSize(e.Delta);

    private void CropCanvasHost_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (sender is System.Windows.FrameworkElement host)
            _viewModel.Crop.Relayout(host.ActualWidth, host.ActualHeight);
    }

    private void CropTab_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        // Don't hijack keystrokes aimed at an input control (caption-style boxes, the size combo, ...).
        // Enter on a Button must still activate that button, so skip buttons too.
        if (Keyboard.FocusedElement is System.Windows.Controls.TextBox
            || Keyboard.FocusedElement is System.Windows.Controls.ComboBox
            || Keyboard.FocusedElement is System.Windows.Controls.Button) return;
        if (Keyboard.Modifiers != ModifierKeys.None) return;
        
        if (e.Key == Key.A)
        {
            _viewModel.Crop.PrevCommand.Execute(null);
            e.Handled = true;
        }
        else if (e.Key == Key.D)
        {
            _viewModel.Crop.NextCommand.Execute(null);
            e.Handled = true;
        }
        else if (e.Key == Key.Space)
        {
            // Add the saved crops to the Process queue.
            _viewModel.Crop.AddCropsToQueueCommand.Execute(null);
            e.Handled = true;
        }
    }

    private void OutputList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (sender is System.Windows.Controls.ListBox list)
            _viewModel.Output.SyncSelection(list.SelectedItems.Cast<OutputItemViewModel>());
    }

    private void OutputTab_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        // Don't hijack keystrokes aimed at input controls: typing in the caption box,
        // the filter combo (Enter confirms a filter pick), or a button (Enter clicks it).
        if (Keyboard.FocusedElement is System.Windows.Controls.TextBox
            or System.Windows.Controls.ComboBox
            or System.Windows.Controls.Button) return;
        if (Keyboard.Modifiers != ModifierKeys.None) return;

        if (e.Key == Key.Delete)
        {
            _viewModel.Output.DeleteSelectedCommand.Execute(null);
            e.Handled = true;
        }
        else if (e.Key == Key.Enter)
        {
            _viewModel.Output.ConfirmAndAdvanceCommand.Execute(null);
            e.Handled = true;
        }
        else
        {
            int? folder = e.Key switch
            {
                >= Key.D1 and <= Key.D9 => e.Key - Key.D1 + 1,
                >= Key.NumPad1 and <= Key.NumPad9 => e.Key - Key.NumPad1 + 1,
                _ => null
            };
            if (folder is not null)
            {
                _viewModel.Output.MoveSelectedTo(folder.Value);
                e.Handled = true;
            }
        }
    }
}
