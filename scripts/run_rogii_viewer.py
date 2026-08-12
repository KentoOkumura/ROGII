"""Launch Tom's ROGII PySide viewer against this repository's data layout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from config_utils import load_project_config, project_path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = REPO_ROOT / "tools" / "rogii-viewer"
DEFAULT_DATASET = project_path(load_project_config(), "data.raw_dir")


def running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset folder containing train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional Kaggle-format predictions CSV with id,tvt columns.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a headless construction/load/render check and exit.",
    )
    parser.add_argument(
        "--platform",
        choices=["auto", "xcb", "wayland", "wayland-egl", "vnc", "minimal", "offscreen"],
        default="auto",
        help="Qt platform backend. WSL usually works best with xcb or wayland.",
    )
    return parser.parse_args()


def configure_qt(smoke_test: bool, platform: str) -> None:
    if smoke_test:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    elif platform != "auto":
        os.environ["QT_QPA_PLATFORM"] = platform
    elif "QT_QPA_PLATFORM" not in os.environ and running_under_wsl():
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    elif os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

    try:
        import PySide6
    except ImportError:
        return

    plugin_path = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_path))


def require_viewer_source() -> None:
    if not (VIEWER_ROOT / "viewer" / "__main__.py").is_file():
        raise SystemExit(
            "ROGII viewer source is missing. Expected it at "
            f"{VIEWER_ROOT}. Clone https://github.com/tom99763/rogii-viewer there."
        )
    sys.path.insert(0, str(VIEWER_ROOT))


def load_dataset(window, dataset: Path) -> None:
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset}")
    window.open_dataset(dataset)
    if window.dataset is None or not window.dataset.wells:
        raise SystemExit(f"No wells found under dataset folder: {dataset}")
    n_train = sum(1 for well in window.dataset.wells if well["split"] == "train")
    n_test = sum(1 for well in window.dataset.wells if well["split"] == "test")
    print(
        f"Loaded dataset: {n_train} train wells + {n_test} test wells from {dataset}",
        flush=True,
    )


def load_predictions(window, predictions: Path | None) -> None:
    if predictions is None:
        return

    from viewer.data import Predictions

    predictions = predictions.resolve()
    if not predictions.is_file():
        raise SystemExit(f"Predictions CSV not found: {predictions}")
    window.predictions = Predictions.load(predictions)
    window.info_panel.set_predictions_status(predictions.name, window.predictions.coverage())


def run_smoke_test(window, output: Path = Path("/tmp/rogii_viewer_smoke.png")) -> None:
    train_entry = next((w for w in window.dataset.wells if w["split"] == "train"), None)
    if train_entry is None:
        raise SystemExit("No train wells found for smoke test.")

    window._on_well_selected(f"train/{train_entry['well_id']}")
    if window.current_well is None:
        raise SystemExit("Smoke test failed: no current well after selection.")

    pix = window.cross_section.grab()
    if pix.isNull():
        raise SystemExit("Smoke test failed: cross-section grab returned a null pixmap.")
    pix.save(str(output), "PNG")
    if output.stat().st_size <= 0:
        raise SystemExit(f"Smoke test failed: empty PNG output at {output}")

    print(
        "viewer_smoke_ok "
        f"well={window.current_well.well_id} rows={window.current_well.n_rows} "
        f"png={output} bytes={output.stat().st_size}"
    )


def main() -> int:
    args = parse_args()
    configure_qt(args.smoke_test, args.platform)
    require_viewer_source()

    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from viewer.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    load_dataset(window, args.dataset)
    load_predictions(window, args.predictions)

    if args.smoke_test:
        run_smoke_test(window)
        return 0

    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    window.resize(1400, 900)
    window.move(80, 80)
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    window.setWindowFlag(Qt.WindowStaysOnTopHint, False)
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    print(
        "ROGII viewer is running "
        f"(QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', 'auto')}, "
        f"DISPLAY={os.environ.get('DISPLAY', '')}, "
        f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}). "
        "Close the viewer window to return to the shell.",
        flush=True,
    )
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
