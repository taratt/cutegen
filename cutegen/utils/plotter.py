import os
import json
import matplotlib.pyplot as plt
import numpy as np

def process_directory(subdir_path, plots_dir):
    depths = []
    ratios = []

    # Loop over all JSON files inside this subdir
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

        # Only process if error_type == PASS
        if data.get("error_type") != "PASS":
            continue

        depth = data.get("depth")
        time_mean = data.get("time", {}).get("mean")
        ref_mean = data.get("ref_time", {}).get("mean")

        if depth is not None and time_mean and ref_mean and ref_mean > 0:
            ratio = ref_mean / time_mean
            depths.append(depth)
            ratios.append(ratio)

    # Nothing valid in this directory
    if not depths:
        return

    # Sort by depth for nicer plotting
    sorted_pairs = sorted(zip(depths, ratios), key=lambda x: x[0])
    depths, ratios = zip(*sorted_pairs)

    # Bar plot
    plt.figure(figsize=(7,5))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(depths)))  # pastel colormap
    plt.bar(depths, ratios, color=colors, edgecolor="black")

    plt.xlabel("Depth")
    plt.ylabel("Speedup (reference time/ generated kernel time)")
    plt.title(f"Speedup vs Depth")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Make sure x-axis shows only the depths we have
    plt.xticks(depths)
    # Save
    os.makedirs(plots_dir, exist_ok=True)
    base_name = os.path.basename(subdir_path.rstrip("/"))
    out_path = os.path.join(plots_dir, "1"+f"{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def process_all(main_dir, plots_dir="plots"):
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.isdir(subdir_path):
            process_directory(subdir_path, plots_dir)


if __name__ == "__main__":
    main_dir = ""  # your root dir
    process_all(main_dir, plots_dir="../plots")
