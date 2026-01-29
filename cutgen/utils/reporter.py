import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

ERRORS = {}
def process_directory(subdir_path):
    """
    Process one subdir -> return a dict: {depth: Counter(error_types)}
    """
    depth_errors = defaultdict(Counter)

    for fname in os.listdir(subdir_path):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(subdir_path, fname)

        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {fpath}, failed to load: {e}")
            continue

        depth = data.get("depth")
        if depth is None:
            continue

        # include top-level error_type
        top_err = data.get("error_type", "NONE")
        depth_errors[depth][top_err] += 1

        # include history errors
        for h in data.get("history", []):
            herr = h.get("error_type", "NONE")
            depth_errors[depth][herr] += 1

    return depth_errors


def visualize_errors(main_dir, plots_dir="plots_errors"):
    os.makedirs(plots_dir, exist_ok=True)
    compile_to_pass = 0
    compile_to_correct = 0
    compile_total = 0
    correct_to_pass = 0
    correct_total = 0
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        depth_errors = process_directory(subdir_path)
        if not depth_errors:
            continue

        # Collect unique error types
        all_error_types = sorted({err for counter in depth_errors.values() for err in counter.keys()})

        depths = sorted(depth_errors.keys())
        error_counts = {err: [depth_errors[d].get(err, 0) for d in depths] for err in all_error_types}

        # --- Report ---
        print(f"\n📊 Report for {subdir}:")
        for d in depths:
            print(f"  Depth {d}: {dict(depth_errors[d])}")

        # --- Visualization ---
        if len(depths) == 1:
            bar_width = 0.3   # thinner if only one bar
        else:
            bar_width = 0.6

        bottoms = np.zeros(len(depths))
        plt.figure(figsize=(8,6))

        pastel = plt.cm.Pastel1(np.linspace(0, 1, len(all_error_types)))  # pastel colors

        for idx, err in enumerate(all_error_types):
            counts = error_counts[err]
            plt.bar(depths, counts, bottom=bottoms,
                    width=bar_width,               # <<< width control here
                    label=err,
                    color=pastel[idx % len(pastel)],
                    edgecolor="black")
            bottoms += np.array(counts)

        plt.xlabel("Depth")
        plt.ylabel("Count of Errors")
        plt.title(f"Error Types by Depth\n{subdir}")
        plt.xticks(depths)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = os.path.join(plots_dir, f"{subdir}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved plot: {out_path}")

        # --- Transition analysis per JSON in this subdir ---
        for fname in os.listdir(subdir_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(subdir_path, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            history = data.get("history", [])
            final_err = data.get("error_type")

            # --- Transition analysis per JSON in this subdir ---
            for fname in os.listdir(subdir_path):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(subdir_path, fname)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                history = data.get("history", [])
                final_err = data.get("error_type")

                # COMPILE transitions
                if any(h.get("error_type") == "COMPILE" for h in history):
                    compile_total += 1
                    if final_err == "PASS":
                        compile_to_pass += 1
                    elif final_err == "CORRECT":
                        compile_to_correct += 1

                # CORRECT transitions
                if any(h.get("error_type") == "CORRECT" for h in history):
                    correct_total += 1
                    if final_err == "PASS":
                        correct_to_pass += 1

        # --- Overall transition report ---
        print("\n====== 🔄 Transition Report ======")
        if compile_total > 0:
            print(f"COMPILE → PASS: {compile_to_pass}/{compile_total} "
                  f"({100 * compile_to_pass / compile_total:.1f}%)")
            print(f"COMPILE → CORRECT: {compile_to_correct}/{compile_total} "
                  f"({100 * compile_to_correct / compile_total:.1f}%)")
        else:
            print("No COMPILE transitions found.")

        if correct_total > 0:
            print(f"CORRECT → PASS: {correct_to_pass}/{correct_total} "
                  f"({100 * correct_to_pass / correct_total:.1f}%)")
        else:
            print("No CORRECT transitions found.")


if __name__ == "__main__":
    main_dir = "/home/tarasaba/PycharmProjects/cutgen/saved_nodes/cute/attempt_4"  # adjust path
    visualize_errors(main_dir, plots_dir="../cute_plots_errors4")
