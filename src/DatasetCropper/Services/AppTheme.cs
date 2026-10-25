using System.Windows;
using System.Windows.Threading;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;

namespace DatasetCropper.Services;

/// <summary>
/// Detects the Windows light/dark setting (HKCU\...\Personalize\AppsUseLightTheme)
/// and applies the matching Material Design theme to the whole application.
/// A lightweight 2 s poll keeps it in sync when the user flips the setting
/// live — WPF does not raise an event for theme changes on its own.
/// </summary>
public static class AppTheme
{
    /// <summary>
    /// True when the system is in dark mode (also the safe default when the
    /// setting cannot be read, e.g. Windows 11 with no prior choice).
    /// </summary>
    public static bool IsDark { get; private set; } = true;

    private static ResourceDictionary? _bundled;

    private static readonly DispatcherTimer Watcher = new()
    {
        Interval = TimeSpan.FromSeconds(2),
    };

    /// <summary>
    /// Reads the current system theme setting and (re)applies the matching
    /// Material Design dictionaries on the running Application. Idempotent;
    /// only swaps the color dictionary when the setting actually changed, so
    /// the existing control styles (which reference the brushes via dynamic
    /// resource lookup) refresh automatically.
    /// </summary>
    public static void Apply()
    {
        bool dark = DetectSystemDark();
        if (dark == IsDark && _bundled is not null) return;
        IsDark = dark;

        var merged = Application.Current.Resources.MergedDictionaries;
        if (_bundled is not null && merged.Contains(_bundled)) merged.Remove(_bundled);

        // BundledTheme is itself a ResourceDictionary that exposes every
        // brush + control style needed for Material Design.  BaseTheme picks
        // the light/dark paper / text / card colors from the stock Material
        // palette.
        _bundled = new BundledTheme
        {
            BaseTheme = dark ? BaseTheme.Dark : BaseTheme.Light,
        };
        merged.Add(_bundled);
    }

    /// <summary>
    /// Starts the background poll that keeps the theme in sync with the
    /// system light/dark setting while the app is running.
    /// </summary>
    public static void StartWatcher()
    {
        if (Watcher.IsEnabled) return;
        Watcher.Tick += (_, _) =>
        {
            if (DetectSystemDark() != IsDark) Apply();
        };
        Watcher.Start();
    }

    private static bool DetectSystemDark()
    {
        // AppsUseLightTheme is 1 in light mode, 0 in dark mode.
        // If the value is absent (very early Windows 11 installs), dark is
        // the safer default to match the system's own dark default.
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(
                @"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
            if (key?.GetValue("AppsUseLightTheme") is int value)
                return value != 1;
            return true;
        }
        catch
        {
            // No read access (locked-down profile, etc.) — treat as dark.
            return true;
        }
    }
}
