"""Decode the edited videos and verify delivery properties; generate a QA contact sheet."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from build_video import ROOT, find_ffmpeg, frame_to_png, sha256


def inspect(ffmpeg: str, path: Path) -> dict:
    result = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0",
                             "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Decode failed: {path}\n{result.stderr}")
    frames = re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)
    video_line = next(line for line in result.stderr.splitlines() if "Video:" in line)
    dimensions = re.search(r",\s+(\d{2,5})x(\d{2,5})(?:[\s,\[])", video_line)
    fps = re.search(r"\b([\d.]+) fps\b", video_line)
    return {"frames": int(frames[-1]), "width": int(dimensions[1]), "height": int(dimensions[2]),
            "fps": float(fps[1]), "h264": "Video: h264" in video_line,
            "yuv420p": "yuv420p" in video_line, "audio": "Audio:" in result.stderr,
            "decode_ok": True, "sha256": sha256(path)}


def verify(root: Path, ffmpeg: str) -> dict:
    timeline = json.loads((root / "media/edit/timeline.json").read_text(encoding="utf-8"))
    checks = []
    targets = [*timeline["sequence"], {"path": timeline["master"], "frames": timeline["frames"]}]
    for target in targets:
        measured = inspect(ffmpeg, root / target["path"])
        assert measured["frames"] == target["frames"], (target, measured)
        assert (measured["width"], measured["height"], measured["fps"]) == (1920, 1080, 30)
        assert measured["h264"] and measured["yuv420p"] and not measured["audio"]
        checks.append(dict(path=target["path"], **measured))
    assert sum(x["frames"] for x in timeline["sequence"]) == timeline["frames"] == 2401
    assert timeline["source_motion_seconds"] == 60
    # Also decode all originals, checking the declared native frame counts.
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    sources = []
    for episode in manifest["episodes"]:
        for camera, expected in episode["source_videos"].items():
            relative = Path("media/source") / episode["case_id"] / f"{camera}.mp4"
            measured = inspect(ffmpeg, root / relative)
            assert measured["sha256"] == expected["sha256"]
            assert measured["frames"] == expected["frames"]
            assert (measured["width"], measured["height"], measured["fps"]) == (128, 128, 20)
            sources.append(dict(path=relative.as_posix(), **measured))

    # This is a QA sheet only; labels never enter the delivery videos.
    from PIL import Image, ImageDraw
    qa = root / "tmp/video-qa"
    qa.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 480, 300
    sheet = Image.new("RGB", (cell_w * 3, cell_h * len(timeline["sequence"])), "#111923")
    draw = ImageDraw.Draw(sheet)
    for row, target in enumerate(timeline["sequence"]):
        for col, frame in enumerate((0, target["frames"] // 2, target["frames"] - 1)):
            png = qa / f"{row:02}_{col}.png"
            frame_to_png(ffmpeg, root / target["path"], frame, png)
            with Image.open(png) as frame_image:
                sheet.paste(frame_image.resize((480, 270)), (col * cell_w, row * cell_h + 25))
            draw.text((col * cell_w + 8, row * cell_h + 5), f"{row:02} {target['case_id']} / frame {frame}", fill="white")
    contact = root / "media/edit/contact_sheet.jpg"
    sheet.save(contact, quality=92)
    report = {"status": "PASS", "master_duration_seconds": timeline["frames"] / 30,
              "source_motion_seconds": 60, "master_frames": timeline["frames"],
              "videos": checks, "source_videos": sources,
              "contact_sheet": contact.relative_to(root).as_posix(),
              "visual_review": "Contact sheet requires human/agent visual inspection; decode checks alone do not assess content."}
    (root / "media/video_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "edited_videos": len(checks), "sources": len(sources),
                      "duration_seconds": timeline["frames"] / 30, "contact_sheet": str(contact)}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    verify(args.root.resolve(), find_ffmpeg(args.ffmpeg))


if __name__ == "__main__":
    main()
