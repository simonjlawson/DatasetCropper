namespace DatasetCropper.Models;

/// <summary>
/// Which concept-detection model the pipeline uses.
/// <see cref="GroundingDino"/> is open-vocabulary (the default): it can detect
/// *any* concept you type. <see cref="Yolo"/> is a fast, small YOLOv8/YOLO11
/// (COCO-80 class) detector: far quicker and lighter, but it only understands its
/// fixed class set, so your concepts are mapped to their nearest COCO class and a
/// concept with no COCO analogue (e.g. "face", "sailboat") won't be found by it.
/// </summary>
public enum DetectorModel
{
    GroundingDino,
    Yolo,
}
