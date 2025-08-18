# export_release.py

import os, re, json, ast, argparse, shutil, hashlib
from pathlib import Path
from glob import glob

DEFAULT_TAGS = ["001","010","100","011","101","110","111"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_txt_config(path: Path) -> dict:
    cfg = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                # numbers, lists, dicts, booleans, quoted strings
                cfg[k] = ast.literal_eval(v)
            except Exception:
                # fallback to plain string without quotes
                cfg[k] = v.strip('"').strip("'")
    return cfg

def newest_dir(paths):
    return max(paths, key=lambda p: Path(p).stat().st_mtime)

def find_source_dir_for_tag(src_root: Path, tag: str) -> Path | None:
    # typical pattern: {tag}_trial_{trial_number}
    candidates = glob(str(src_root / f"{tag}_trial_*"))
    if not candidates:
        return None
    return Path(newest_dir(candidates))

def pick_one(path_glob: str) -> Path | None:
    hits = [Path(p) for p in glob(path_glob)]
    if not hits:
        return None
    # prefer files that include 'decoder_ckpt' or '<tag>_config'
    hits.sort(key=lambda p: p.name)
    return hits[0]

def export_one_tag(tag: str, src_root: Path, dst_models: Path, stats_dir: Path | None, copy_stats: bool, manifest: dict):
    src_dir = find_source_dir_for_tag(src_root, tag)
    if src_dir is None:
        print(f"⚠️  Skipping {tag}: no folder like {tag}_trial_* under {src_root}")
        return

    # locate files in source
    cfg_txt = pick_one(str(src_dir / f"*{tag}*_config_*.txt")) or pick_one(str(src_dir / f"*config*.txt"))
    dec_pt  = pick_one(str(src_dir / "decoder_ckpt*.pt")) or pick_one(str(src_dir / "decoder*.pt"))

    if cfg_txt is None or dec_pt is None:
        print(f"⚠️  Skipping {tag}: missing files (config txt: {cfg_txt is not None}, decoder pt: {dec_pt is not None}) in {src_dir}")
        return

    # destination folder
    dst_dir = dst_models / tag
    dst_dir.mkdir(parents=True, exist_ok=True)
    cfg_json_path = dst_dir / f"config_{tag}.json"
    dec_out_path  = dst_dir / f"decoder_{tag}.pt"

    # convert config txt -> json
    cfg = parse_txt_config(cfg_txt)
    with open(cfg_json_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # copy & rename decoder
    shutil.copy2(dec_pt, dec_out_path)

    # optional: copy stats
    stats = {}
    if copy_stats and stats_dir is not None:
        pm_src = stats_dir / f"P_mean_theta_{tag}.npy"
        ps_src = stats_dir / f"P_std_theta_{tag}.npy"
        sm_src = stats_dir / f"S_mean_theta_{tag}.npy"
        ss_src = stats_dir / f"S_std_theta_{tag}.npy"

        pm_dst = dst_dir / f"P_mean_{tag}.npy"
        ps_dst = dst_dir / f"P_std_{tag}.npy"
        sm_dst = dst_dir / f"S_mean_{tag}.npy"
        ss_dst = dst_dir / f"S_std_{tag}.npy"

        for src, dst in [(pm_src, pm_dst), (ps_src, ps_dst), (sm_src, sm_dst), (ss_src, ss_dst)]:
            if src.exists():
                shutil.copy2(src, dst)
                stats[dst.name] = {"path": str(dst), "sha256": sha256(dst)}
            else:
                print(f"⚠️  Stats missing for {tag}: {src.name}")

    # add to manifest
    entry = {
        "tag": tag,
        "source_dir": str(src_dir),
        "files": {
            "config":  {"path": str(cfg_json_path), "sha256": sha256(cfg_json_path)},
            "decoder": {"path": str(dec_out_path),  "sha256": sha256(dec_out_path)},
            **stats,
        }
    }
    manifest.setdefault("models", []).append(entry)
    print(f"✅ Exported {tag} → {dst_dir}")

def main():
    ap = argparse.ArgumentParser(description="Export best checkpoints into release_v1/models with JSON configs.")
    ap.add_argument("--src",   type=str, default="checkpoints/best", help="Folder containing <tag>_trial_* dirs.")
    ap.add_argument("--dst",   type=str, default="release_v1/models", help="Destination models folder.")
    ap.add_argument("--stats", type=str, default="data/partition_by_theta", help="Folder with *_theta_<tag>.npy stats.")
    ap.add_argument("--tags",  type=str, default=",".join(DEFAULT_TAGS), help="Comma-separated tags (e.g., 001,010,...).")
    ap.add_argument("--no-stats", action="store_true", help="Do not copy P/S mean/std files.")
    args = ap.parse_args()

    src_root   = Path(args.src)
    dst_models = Path(args.dst)
    release_root = dst_models.parent  # e.g., release_v1/
    stats_dir  = None if args.no_stats or not args.stats else Path(args.stats)

    dst_models.mkdir(parents=True, exist_ok=True)
    manifest = {"source_root": str(src_root), "dst_models": str(dst_models), "tags": []}

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    manifest["tags"] = tags

    for tag in tags:
        export_one_tag(tag, src_root, dst_models, stats_dir, copy_stats=not args.no_stats, manifest=manifest)

    # write manifest at release root
    (release_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n📄 Manifest written to {release_root / 'manifest.json'}")

if __name__ == "__main__":
    main()
