"""Execute the notebook headless with per-cell progress logging.

Usage: python run_notebook.py notebooks/stock_price_prediction.ipynb
"""
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class LoggingClient(NotebookClient):
    def on_cell_start(self, cell, cell_index):
        first = next((ln.strip() for ln in cell.source.splitlines() if ln.strip()), "")
        print(f"  cell {cell_index:02d} start  {first[:75]}", flush=True)

    def on_cell_complete(self, cell, cell_index):
        print(f"  cell {cell_index:02d} done   ({time.strftime('%H:%M:%S')})", flush=True)


def main(path: str) -> int:
    nb_path = Path(path)
    nb = nbformat.read(nb_path, as_version=4)
    client = LoggingClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    started = time.time()
    try:
        client.execute()
    except CellExecutionError:
        print("\nEXECUTION FAILED\n", flush=True)
        traceback.print_exc()
        nbformat.write(nb, nb_path.with_name("_failed_" + nb_path.name))
        print(f"Partial notebook written to notebooks/_failed_{nb_path.name}", flush=True)
        return 1
    except Exception:
        print("\nEXECUTION ERROR\n", flush=True)
        traceback.print_exc()
        return 1

    nbformat.write(nb, nb_path)
    print(f"\nOK: executed {len(nb.cells)} cells in {time.time() - started:.1f}s -> {nb_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
