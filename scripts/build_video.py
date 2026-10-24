"""Build a text-free, 1080p edit from the eight supplied two-camera recordings.

Requires FFmpeg with libx264, or the imageio-ffmpeg Python package.
No model inference, frame interpolation, captions, or audio is added.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
          "-r", str(FPS), "-video_track_timescale", "15360", "-movflags", "+faststart", "-an"]


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(explicit)
        return str(Path(explicit).resolve())
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit("Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg PATH.") from exc


def run(ffmpeg: str, arguments: list[str]) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timecode(frame: int) -> str:
    seconds, frames = divmod(frame, FPS)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}:{frames:02}"


def frame_to_png(ffmpeg: str, video: Path, frame: int, target: Path) -> None:
    run(ffmpeg, ["-i", str(video), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-update", "1", str(target)])


def build(root: Path, ffmpeg: str) -> dict:
    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    edit = root / "media" / "edit"
    output = root / "media" / "output"
    temporary = root / "tmp" / "video-build"
    for folder in (edit, output, temporary):
        folder.mkdir(parents=True, exist_ok=True)
    clips = []
    source_elapsed = 0.0
    for index, episode in enumerate(episodes, 1):
        case = episode["case_id"]
        source = root / "media" / "source" / case
        for camera in ("agentview", "wrist"):
            path = source / f"{camera}.mp4"
            if sha256(path) != episode["source_videos"][camera]["sha256"]:
                raise ValueError(f"Source video hash mismatch: {case}/{camera}.mp4")
        source_elapsed += episode["duration_seconds"]
        native = episode["source_videos"]["agentview"]
        frame_count = math.ceil(Fraction(native["frames"], native["fps"]) * FPS)
        target = edit / f"{index:02}_{case}.mp4"
        # Round upward per clip so no source duration is trimmed at the 20-to-30 fps
        # boundary. The selected clips total 60 native seconds and 60 + 1/30 edited seconds.
        filters = (
            "[0:v]fps=30,tpad=stop_mode=clone:stop_duration=1,scale=768:768:flags=lanczos,setsar=1[a];"
            "[1:v]fps=30,tpad=stop_mode=clone:stop_duration=1,scale=512:512:flags=lanczos,setsar=1[w];"
            "color=c=0x111923:s=1920x1080:r=30[bg];"
            "[bg][a]overlay=240:120:shortest=1[mid];"
            "[mid][w]overlay=1168:248:shortest=1,format=yuv420p[out]"
        )
        run(ffmpeg, ["-i", str(source / "agentview.mp4"), "-i", str(source / "wrist.mp4"),
                     "-filter_complex", filters, "-map", "[out]", "-frames:v", str(frame_count),
                     *ENCODE, str(target)])
        clips.append({"case_id": case, "title_ja": episode["title_ja"],
                      "frames": frame_count, "source_duration_seconds": episode["duration_seconds"],
                      "path": target.relative_to(root).as_posix()})
        print(f"Built {target.name}: {frame_count} frames", flush=True)

    intro_image = temporary / "intro.png"
    outro_image = temporary / "outro.png"
    frame_to_png(ffmpeg, root / clips[0]["path"], 0, intro_image)
    # End on the final successful case; the failure itself remains complete in the edit.
    closing_clip = clips[-2] if len(clips) > 1 else clips[-1]
    frame_to_png(ffmpeg, root / closing_clip["path"], closing_clip["frames"] - 1, outro_image)
    cards = []
    for name, still in (("00_intro", intro_image), ("99_outro", outro_image)):
        target = edit / f"{name}.mp4"
        run(ffmpeg, ["-loop", "1", "-framerate", str(FPS), "-i", str(still),
                     "-frames:v", str(10 * FPS), *ENCODE, str(target)])
        cards.append({"case_id": name, "title_ja": "導入用静止画" if name == "00_intro" else "まとめ用静止画",
                      "frames": 10 * FPS, "source_duration_seconds": 0,
                      "path": target.relative_to(root).as_posix()})
    sequence = [cards[0], *clips, cards[1]]
    # All filenames are generated above and contain no quotes or line breaks.
    concat_file = edit / "concat.txt"
    concat_file.write_text("".join(f"file '{Path(item['path']).name}'\n" for item in sequence), encoding="utf-8")
    master = output / "pai_demo_no_subtitles.mp4"
    run(ffmpeg, ["-f", "concat", "-safe", "1", "-i", str(concat_file), "-c", "copy", "-an",
                 "-movflags", "+faststart", str(master)])
    current = 0
    rows = []
    for item in sequence:
        row = dict(item, start_frame=current, end_frame_exclusive=current + item["frames"],
                   start_seconds=round(current / FPS, 6), end_seconds=round((current + item["frames"]) / FPS, 6),
                   start_timecode=timecode(current), end_timecode_exclusive=timecode(current + item["frames"]))
        rows.append(row)
        current += item["frames"]
    with (edit / "timeline.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"schema_version": 1, "width": 1920, "height": 1080, "fps": FPS,
              "codec": "H.264", "pixel_format": "yuv420p", "audio": False, "captions": False,
              "source_manifest_sha256": sha256(manifest_path), "source_motion_seconds": source_elapsed,
              "duration_seconds": current / FPS, "frames": current,
              "layout": {"background": "#111923", "agentview": [240, 120, 768, 768],
                         "wrist": [1168, 248, 512, 512], "suggested_caption_band": [160, 920, 1600, 120]},
              "master": master.relative_to(root).as_posix(), "master_sha256": sha256(master), "sequence": rows}
    (edit / "timeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Master: {master} ({current / FPS:.2f} s)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    build(args.root.resolve(), find_ffmpeg(args.ffmpeg))


if __name__ == "__main__":
    main()
