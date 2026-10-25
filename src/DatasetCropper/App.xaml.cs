using System.Windows;
using DatasetCropper.Services;

namespace DatasetCropper;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Apply the Material Design theme (light/dark based on the system's
        // setting) BEFORE the main window is created so the first frame is
        // already styled, then keep it in sync with the system's setting.
        AppTheme.Apply();
        AppTheme.StartWatcher();

        // Global last-chance handler so an unhandled worker-thread exception
        // surfaces in the log instead of silently killing the process.
        DispatcherUnhandledException += (_, args) =>
        {
            string msg = "Unhandled UI exception: " + args.Exception;
            System.Diagnostics.Debug.WriteLine(msg);
            MessageBox.Show(msg, "DatasetCropper", MessageBoxButton.OK, MessageBoxImage.Error);
            args.Handled = true;
        };
    }
}
