"""Check that a clone or ZIP copy contains the documented local inputs.

Uses only Python's standard library. This checks files, recorded checksums and
local Markdown links; it does not install dependencies or run AI/simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ".gitattributes", ".gitignore", "README.md", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "requirements-video.txt", "data/manifest.json",
    "data/model_training.json", "data/task_sources.json",
    "docs/editing.md", "docs/reproduction.md", "docs/provenance.md",
    "docs/validation.md", "docs/model.md", "docs/edit_timeline.csv", "docs/video_build.json",
    "docs/video_verification.json", "scripts/build_video.py",
    "scripts/verify_video.py", "scripts/check_runtime.py",
    "scripts/check_repository.py", "src/pai_demo/__init__.py",
    "src/pai_demo/__main__.py", "src/pai_demo/episode.py",
    "src/pai_demo/rollout.py", "src/pai_demo/policy.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(root: Path) -> dict:
    root = root.resolve()
    errors = []
    counts = {"required_files": len(REQUIRED), "episodes": 0,
              "source_videos": 0, "checksum_files": 0, "local_links": 0,
              "task_definitions": 0, "task_cases": 0}

    def error(path, issue):
        errors.append({"path": str(path), "issue": issue})

    def read_json(relative):
        path = root / relative
        if not path.is_file():
            error(relative, "missing file")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            error(relative, "cannot read valid UTF-8 JSON")
            return None

    def check_hash(relative, expected):
        path = root / relative
        if not path.is_file():
            error(relative, "missing file")
            return False
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            error(relative, "invalid expected SHA-256")
            return False
        try:
            actual = sha256(path)
        except OSError:
            error(relative, "cannot read file")
            return False
        counts["checksum_files"] += 1
        if actual != expected:
            error(relative, "SHA-256 mismatch")
            return False
        return True

    for relative in REQUIRED:
        if not (root / relative).is_file():
            error(relative, "missing required file")

    manifest = read_json("data/manifest.json")
    episodes = manifest.get("episodes", []) if isinstance(manifest, dict) else []
    if not isinstance(episodes, list):
        error("data/manifest.json", "episodes must be a list")
        episodes = []
    if len(episodes) != 8:
        error("data/manifest.json", "expected 8 selected episodes")
    seen = set()
    metadata_by_case = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            error("data/manifest.json", "episode must be an object")
            continue
        case = episode.get("case_id")
        if not isinstance(case, str) or not re.fullmatch(r"[a-z0-9_]+", case):
            error("data/manifest.json", "invalid case_id")
            continue
        if case in seen:
            error("data/manifest.json", f"duplicate case_id: {case}")
            continue
        seen.add(case)
        counts["episodes"] += 1
        folder = f"data/episodes/{case}"
        if episode.get("episode_dir") != folder:
            error("data/manifest.json", f"invalid episode_dir for {case}")
        check_hash(f"{folder}/metadata.json", episode.get("metadata_sha256"))
        trajectory = f"{folder}/trajectory.npz"
        if check_hash(trajectory, episode.get("trajectory_sha256")):
            try:
                with zipfile.ZipFile(root / trajectory) as archive:
                    if archive.testzip() is not None:
                        error(trajectory, "NPZ archive CRC mismatch")
                    expected = {"actions.npy", "initial_state.npy", "pre_joint.npy", "post_joint.npy"}
                    if set(archive.namelist()) != expected:
                        error(trajectory, "unexpected or missing NPZ arrays")
            except (OSError, zipfile.BadZipFile):
                error(trajectory, "invalid NPZ archive")
        metadata = read_json(f"{folder}/metadata.json")
        if isinstance(metadata, dict):
            metadata_by_case[case] = metadata
            if metadata.get("case_id") != case:
                error(f"{folder}/metadata.json", "case_id differs from manifest")
            if metadata.get("trajectory_sha256") != episode.get("trajectory_sha256"):
                error(f"{folder}/metadata.json", "trajectory checksum differs from manifest")
        cameras = episode.get("source_videos", {})
        if not isinstance(cameras, dict) or set(cameras) != {"agentview", "wrist"}:
            error("data/manifest.json", f"expected two cameras for {case}")
            continue
        for camera in ("agentview", "wrist"):
            relative = f"media/source/{case}/{camera}.mp4"
            source = cameras[camera]
            expected_hash = source.get("sha256") if isinstance(source, dict) else None
            if check_hash(relative, expected_hash):
                counts["source_videos"] += 1
            if isinstance(metadata, dict) and metadata.get("source_videos", {}).get(camera) != source:
                error(f"{folder}/metadata.json", f"{camera} specification differs from manifest")

    training = read_json("data/model_training.json")
    identity = training.get("identity", {}) if isinstance(training, dict) else {}
    recorded_identity = identity.get("source_model_identity_sha256") if isinstance(identity, dict) else None
    expected_identity = manifest.get("source_model_identity_sha256") if isinstance(manifest, dict) else None
    if (not isinstance(recorded_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", recorded_identity)
            or recorded_identity != expected_identity):
        error("data/model_training.json", "source model identity differs from manifest or is invalid")

    task_sources = read_json("data/task_sources.json")
    tasks = task_sources.get("tasks", []) if isinstance(task_sources, dict) else []
    if not isinstance(tasks, list):
        error("data/task_sources.json", "tasks must be a list")
        tasks = []
    if len(tasks) != 7:
        error("data/task_sources.json", "expected 7 public task definitions")
    task_paths, covered_cases = set(), set()
    for task in tasks:
        if not isinstance(task, dict):
            error("data/task_sources.json", "task must be an object")
            continue
        task_path = task.get("bddl_relative_path")
        case_ids = task.get("case_ids")
        if not isinstance(task_path, str) or not task_path:
            error("data/task_sources.json", "missing BDDL path")
            continue
        if task_path in task_paths:
            error("data/task_sources.json", f"duplicate BDDL path: {task_path}")
        task_paths.add(task_path)
        counts["task_definitions"] += 1
        if (not isinstance(case_ids, list) or not case_ids
                or any(not isinstance(case, str) for case in case_ids)):
            error("data/task_sources.json", "case_ids must be a nonempty list of strings")
            continue
        for case in case_ids:
            if case in covered_cases:
                error("data/task_sources.json", f"duplicate task case_id: {case}")
            covered_cases.add(case)
            metadata = metadata_by_case.get(case)
            if metadata is None:
                error("data/task_sources.json", f"unknown or unreadable episode: {case}")
            elif (metadata.get("bddl_relative_path") != task_path
                    or metadata.get("bddl_sha256") != task.get("sha256")):
                error("data/task_sources.json", f"BDDL path or checksum differs for {case}")
        counts["task_cases"] = len(covered_cases)
    for case in sorted(seen - covered_cases):
        error("data/task_sources.json", f"missing episode reference: {case}")

    build_record = read_json("docs/video_build.json")
    if isinstance(build_record, dict):
        check_hash("data/manifest.json", build_record.get("source_manifest_sha256"))

    # Only published documentation is scanned; tmp/ and generated outputs are not.
    markdown = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    for document in markdown:
        relative = document.relative_to(root).as_posix()
        try:
            text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            error(relative, "cannot read UTF-8 Markdown")
            continue
        text = re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text)
        text = re.sub(r"`[^`\n]*`", "", text)
        links = re.findall(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text)
        links += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", text)
        for link in links:
            destination = link.strip("<>")
            if destination.startswith("#"):
                continue
            parsed = urlsplit(destination)
            if parsed.scheme in {"http", "https", "mailto", "tel", "data"} or parsed.netloc:
                continue
            if parsed.scheme or parsed.path.startswith(("/", "\\")):
                error(relative, "non-portable absolute local link")
                continue
            local_path = unquote(parsed.path)
            if not local_path:
                continue
            counts["local_links"] += 1
            target = (document.parent / local_path).resolve()
            if not target.is_relative_to(root):
                error(relative, f"link leaves repository: {local_path}")
            elif not target.exists():
                error(relative, f"missing local link target: {local_path}")

    return {"schema_version": 1, "status": "FAIL" if errors else "PASS",
            "checks": counts, "errors": errors,
            "not_checked": ["external libraries and assets", "model weights",
                            "GPU inference and physics execution", "video decoding"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON report file (overwritten if present)")
    args = parser.parse_args(argv)
    result = check(args.root)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized.encode("utf-8"))
    print(serialized, end="")
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
