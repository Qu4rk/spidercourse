import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path so sibling packages (e.g. `required`) import correctly
# This matters when running the script directly (python helpers/show_best.py) because
# Python sets sys.path[0] to the script's directory (helpers/) which prevents sibling
# top-level packages from being found. Insert the parent directory (project root).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from required.visualiser import visualiseWalk


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
        # Try to auto-find a matching file in output/runs (most GA runs are saved there)
        runs_dir = Path('output') / 'runs'
        if not runs_dir.exists():
            # fallback to workspace search if the runs folder doesn't exist
            candidates = sorted(Path('.').glob('**/best_individual*.txt'))
        else:
            candidates = sorted(runs_dir.glob('best_individual*.txt'))

        if not candidates:
            print('No filename provided and no best_individual*.txt found in output/runs or the workspace.')
            return
        path = candidates[-1]
        print(f"Auto-selected: {path}")
    else:
        path = Path(args.file)

    if not path.exists():
        print(f"File not found: {path}")
        return

    frames = load_individual_from_txt(path)
    # Ensure float dtype and 2D shape for the visualiser
    frames = np.array(frames, dtype=float)

    # visualiseWalk expects frames as an array of shape (num_frames, 24).
    # We loaded `frames` with that shape already, so pass it directly.
    print(f"Loaded individual with shape {frames.shape}; visualising...")
    visualiseWalk(frames)


if __name__ == '__main__':
    main()
