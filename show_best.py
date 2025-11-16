import argparse
from pathlib import Path
import numpy as np
from plot_spider_pose import visualiseWalk


def load_individual_from_txt(path: Path) -> np.ndarray:
    """Load a single best-individual text file produced by GA.save_population_txt.

    File format: one frame per line, comma-separated joint angles (24 values per line).
    Returns a numpy array of shape (frames, joints).
    """
    # Use numpy to load comma-separated floats
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Failed to load '{path}': {e}")

    # If the file contains a single frame, np.loadtxt returns 1D array; normalize to 2D
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr


def main():
    p = argparse.ArgumentParser(description='Load a best_individual text file and visualise the walk')
    p.add_argument('file', nargs='?', default=None, help='Path to the best_individual text file')
    args = p.parse_args()

    if args.file is None:
        # try to auto-find a matching file in the working directory
        cwd = Path('.')
        candidates = sorted(cwd.glob('**/best_individual*.txt')) #change from 26 to wild card
        if not candidates:
            print('No filename provided and no best_individual*.txt found in the workspace.')
            return
        path = candidates[-1]
        print(f"Auto-selected: {path}")
    else:
        path = Path(args.file)

    if not path.exists():
        print(f"File not found: {path}")
        return

    frames = load_individual_from_txt(path)
    #frames = np.array(frames, dtype=float) ## Ensure float type Getting an error without this line

    # visualiseWalk expects frames as an array of shape (num_frames, 24).
    # We loaded `frames` with that shape already, so pass it directly.
    print(f"Loaded individual with shape {frames.shape}; visualising...")
    visualiseWalk(frames)


if __name__ == '__main__':
    main()
