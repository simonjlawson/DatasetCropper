namespace DatasetCropper.Models;

/// <summary>
/// Which mask-generator model the pipeline uses.
/// <see cref="Sam"/> is box-prompted (point-prompted) segmentation &mdash; the
/// default. <see cref="ClipSeg"/> is text-prompted zero-shot segmentation (CLIPSeg
/// UNet: the crop's concept text drives the mask, no box needed).
/// <see cref="RembgHuman"/> is a fixed human/foreground remover (rembg
/// u2net-human-seg; no prompt at all).
/// </summary>
public enum MaskModel
{
    Sam,
    ClipSeg,
    RembgHuman,
}
