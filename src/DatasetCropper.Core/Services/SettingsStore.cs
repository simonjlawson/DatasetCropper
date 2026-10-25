using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using DatasetCropper.Models;

namespace DatasetCropper.Services;

/// <summary>
/// Loads/saves <see cref="AppSettings"/> as JSON under %APPDATA%\DatasetCropper.
/// </summary>
public static class SettingsStore
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public static string SettingsDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DatasetCropper");

    public static string SettingsPath => Path.Combine(SettingsDir, "settings.json");

    /// <summary>Rolling run log, appended to for post-run debugging.</summary>
    public static string LogPath => Path.Combine(SettingsDir, "DatasetCropper.log");

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                var loaded = JsonSerializer.Deserialize<AppSettings>(json, JsonOpts);
                if (loaded is not null) return loaded;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine("Failed to load settings: " + ex);
        }
        return new AppSettings();
    }

    public static void Save(AppSettings settings)
    {
        try
        {
            Directory.CreateDirectory(SettingsDir);
            var json = JsonSerializer.Serialize(settings, JsonOpts);
            File.WriteAllText(SettingsPath, json);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine("Failed to save settings: " + ex);
        }
    }
}
