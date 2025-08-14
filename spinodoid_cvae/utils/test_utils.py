# utils/test_utils.py

def print_checkpoint_summary(TAGS, TRIAL, CKPTS, show_paths=False):
    lines = ["✅ Model checkpoints loaded:"]
    for i, t in enumerate(TAGS):
        line = f"- {t}: trial {TRIAL[t]}"
        if show_paths:
            line += f"  → {CKPTS[i]}"
        lines.append(line)
    print("\n".join(lines))