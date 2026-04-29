import io
import sys
from pathlib import Path

import numpy as np
from cartopy import crs as ccrs
from cartopy.feature import STATES
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QDoubleSpinBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QComboBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset

import wrf
from .colortables import (
    AUTUMNPLOT_JSON_FILES,
    as_float_array,
    convert_autumnplot_levels,
    display_units_name,
    f_to_c,
    f_to_k,
    load_autumnplot_colortable,
    request_units_name,
)
from .constants import (
    BARB_LEVEL_OPTIONS,
    DISPLAY_NAME_OVERRIDES,
    HODO_OVERLAY_COLORS,
    HODO_OVERLAY_LEVELS_HPA,
    HODO_SEGMENT_COLORS,
    PIVOTAL_DBZ_COLORS,
    PIVOTAL_DBZ_LEVELS,
    PIVOTAL_DEWPOINT_COLORS,
    PIVOTAL_DEWPOINT_LEVELS_C,
    PIVOTAL_TEMP_COLORS,
    PIVOTAL_TEMP_LEVELS_F,
    SOUNDING_AXES_FACE,
    SOUNDING_GRID,
    SOUNDING_PANEL_FACE,
    SOUNDING_TEXT,
)
from .metadata import VARIABLE_CATEGORY_ORDER, categorize_variable


CUSTOM_VARIABLE_METADATA = [
    {"name": "qv", "description": "Water Vapor Mixing Ratio", "units": "g/kg"},
    {"name": "qc", "description": "Cloud Water Mixing Ratio", "units": "g/kg"},
    {"name": "qr", "description": "Rain Water Mixing Ratio", "units": "g/kg"},
    {"name": "qi", "description": "Ice Mixing Ratio", "units": "g/kg"},
    {"name": "qs", "description": "Snow Mixing Ratio", "units": "g/kg"},
    {"name": "qg", "description": "Graupel Mixing Ratio", "units": "g/kg"},
    {"name": "QCLOUD", "description": "Cloud Water Mixing Ratio", "units": "g/kg"},
    {"name": "QRAIN", "description": "Rain Water Mixing Ratio", "units": "g/kg"},
    {"name": "QICE", "description": "Ice Mixing Ratio", "units": "g/kg"},
    {"name": "QSNOW", "description": "Snow Mixing Ratio", "units": "g/kg"},
    {"name": "SWDOWN", "description": "Downward Shortwave Flux at Surface", "units": "W/m^2"},
    {"name": "SWNORM", "description": "Normal Incident Shortwave Flux", "units": "W/m^2"},
    {"name": "ALBEDO", "description": "Surface Albedo", "units": "%"},
    {"name": "planetary_albedo", "description": "Planetary Albedo", "units": "%"},
    {"name": "SWDNB", "description": "Downward Shortwave Flux at Bottom", "units": "W/m^2"},
    {"name": "SWDNBC", "description": "Clear-Sky Downward Shortwave Flux at Bottom", "units": "W/m^2"},
    {"name": "SWUPB", "description": "Upward Shortwave Flux at Bottom", "units": "W/m^2"},
    {"name": "SWUPBC", "description": "Clear-Sky Upward Shortwave Flux at Bottom", "units": "W/m^2"},
    {"name": "SWDNT", "description": "Downward Shortwave Flux at Top", "units": "W/m^2"},
    {"name": "SWDNTC", "description": "Clear-Sky Downward Shortwave Flux at Top", "units": "W/m^2"},
    {"name": "SWUPT", "description": "Upward Shortwave Flux at Top", "units": "W/m^2"},
    {"name": "SWUPTC", "description": "Clear-Sky Upward Shortwave Flux at Top", "units": "W/m^2"},
    {"name": "RTHRATSW", "description": "Shortwave Heating Rate", "units": "K/s"},
    {"name": "surface_precip", "description": "Surface Precipitation", "units": "mm"},
]

PBL_VARIABLE_METADATA = [
    {
        "name": "pblh",
        "description": "PBL Depth",
        "units": "m",
        "fetch_names": ["PBLH", "pblh"],
    },
    {
        "name": "tke",
        "description": "Turbulent Kinetic Energy",
        "units": "m2/s2",
        "fetch_names": ["TKE", "tke", "TKE_PBL"],
    },
    {
        "name": "hfx",
        "description": "Sensible Heat Flux",
        "units": "W/m^2",
        "fetch_names": ["HFX", "hfx"],
    },
    {
        "name": "lh",
        "description": "Latent Heat Flux",
        "units": "W/m^2",
        "fetch_names": ["LH", "LHFX", "lh"],
    },
    {
        "name": "qfx",
        "description": "Moisture Flux",
        "units": "kg/m^2/s",
        "fetch_names": ["QFX", "qfx"],
    },
]

PBL_VARIABLES = {item["name"]: item for item in PBL_VARIABLE_METADATA}

TRACK_MAX_VARIABLE_METADATA = [
    {
        "name": "max_stp",
        "description": "Maximum STP Track",
        "units": "unitless",
        "source_names": ["stp", "stp_effective", "stp_fixed"],
    },
    {
        "name": "max_srh1",
        "description": "Maximum 0-1 km SRH Track",
        "units": "m2/s2",
        "source_names": ["srh1"],
    },
    {
        "name": "max_srh3",
        "description": "Maximum 0-3 km SRH Track",
        "units": "m2/s2",
        "source_names": ["srh3"],
    },
]

TRACK_MAX_VARIABLES = {item["name"]: item for item in TRACK_MAX_VARIABLE_METADATA}

MIXING_RATIO_VARIABLES = {
    "qv": "QVAPOR",
    "qc": "QCLOUD",
    "qr": "QRAIN",
    "qi": "QICE",
    "qs": "QSNOW",
    "qg": "QGRAUP",
    "QCLOUD": "QCLOUD",
    "QRAIN": "QRAIN",
    "QICE": "QICE",
    "QSNOW": "QSNOW",
}

DIRECT_DATASET_VARIABLES = {
    "SWDOWN": {"units": "W/m^2"},
    "SWNORM": {"units": "W/m^2"},
    "ALBEDO": {"units": "%"},
    "SWDNB": {"units": "W/m^2"},
    "SWDNBC": {"units": "W/m^2"},
    "SWUPB": {"units": "W/m^2"},
    "SWUPBC": {"units": "W/m^2"},
    "SWDNT": {"units": "W/m^2"},
    "SWDNTC": {"units": "W/m^2"},
    "SWUPT": {"units": "W/m^2"},
    "SWUPTC": {"units": "W/m^2"},
    "RTHRATSW": {"units": "K/s"},
}

DERIVED_ALBEDO_VARIABLES = {"ALBEDO", "planetary_albedo"}

CROSS_SECTION_SOURCE_OVERRIDES = {
    "maxdbz": "dbz",
    "maxdbz_uhel_fill": "dbz",
    "t2": "temp",
    "tv2m": "tv",
    "dp2m": "td",
    "rh2m": "rh",
    "wspd10": "wspd",
    "wdir10": "wdir",
}


def _haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return earth_radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bresenham_indices(j0, i0, j1, i1):
    dj = abs(j1 - j0)
    di = abs(i1 - i0)
    step_j = 1 if j0 < j1 else -1
    step_i = 1 if i0 < i1 else -1
    error = di - dj
    js = [j0]
    is_ = [i0]
    j = j0
    i = i0
    while j != j1 or i != i1:
        double_error = 2 * error
        if double_error > -dj:
            error -= dj
            i += step_i
        if double_error < di:
            error += di
            j += step_j
        js.append(j)
        is_.append(i)
    return np.asarray(js, dtype=int), np.asarray(is_, dtype=int)


class CrossSectionWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.closed_callback = None
        self.setWindowTitle("Vertical Cross Section")
        self.resize(1100, 700)

        container = QWidget()
        layout = QVBoxLayout(container)

        self.info_label = QLabel("Enable the tool, then click start and end points on the map.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(11, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)

        self.setCentralWidget(container)
        self.show_message("Enable the tool, then click start and end points on the map.")

    def show_message(self, message):
        self.info_label.setText(message)
        self.figure.clear()
        axes = self.figure.subplots()
        axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=axes.transAxes)
        axes.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if callable(self.closed_callback):
            self.closed_callback()
        super().closeEvent(event)


class WrfViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WRFOUT Viewer")
        self.resize(1400, 900)

        self.wrf_file = None
        self.wrf_files = []
        self.raw_datasets = []
        self.file_path = None
        self.file_paths = []
        self.times = []
        self.time_sources = []
        self.lat = None
        self.lon = None
        self.variable_metadata = list(wrf.list_variables())
        known_variable_names = {item.get("name") for item in self.variable_metadata}
        for item in CUSTOM_VARIABLE_METADATA + PBL_VARIABLE_METADATA + TRACK_MAX_VARIABLE_METADATA:
            if item["name"] in known_variable_names:
                continue
            self.variable_metadata.append(item)
            known_variable_names.add(item["name"])
        self.variable_metadata.append(
            {
                "name": "maxdbz_uhel_fill",
                "description": "Composite Reflectivity & 1-hr UH > 75 m^2 s^-2",
                "units": "dBZ",
            }
        )
        self.variables_by_category = {}
        self.current_ndim = 0
        self.selected_point = None
        self.cross_section_points = []
        self.cross_section_window = None
        self.map_axes = None
        self._style_range_cache = {}
        self._track_max_cache = {}

        self._build_ui()
        self._populate_variables()
        self.statusBar().showMessage("Open a WRFOUT file to begin.")

    def _build_ui(self):
        root = QSplitter()
        controls = QWidget()
        controls.setMaximumWidth(380)
        controls_layout = QVBoxLayout(controls)

        open_button = QPushButton("Open WRFOUT")
        open_button.clicked.connect(self.open_file)
        controls_layout.addWidget(open_button)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_folder)
        controls_layout.addWidget(open_folder_button)

        self.cross_section_button = QPushButton("Vertical Cross Section")
        self.cross_section_button.setCheckable(True)
        self.cross_section_button.toggled.connect(self.on_cross_section_toggled)
        controls_layout.addWidget(self.cross_section_button)

        form = QFormLayout()

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        form.addRow("File", self.file_label)

        self.category_combo = QComboBox()
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        form.addRow("Category", self.category_combo)

        self.variable_combo = QComboBox()
        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        form.addRow("Variable", self.variable_combo)

        self.units_edit = QLineEdit()
        self.units_edit.setPlaceholderText("Leave blank for default units")
        self.units_edit.editingFinished.connect(self.render_current_time)
        form.addRow("Units", self.units_edit)

        self.level_spin = QSpinBox()
        self.level_spin.setEnabled(False)
        self.level_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Level", self.level_spin)

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setMinimum(0)
        self.smoothing_spin.setMaximum(10)
        self.smoothing_spin.setValue(1)
        self.smoothing_spin.setToolTip("Gaussian smoothing radius in grid cells")
        self.smoothing_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Smooth", self.smoothing_spin)

        self.cross_section_top_spin = QDoubleSpinBox()
        self.cross_section_top_spin.setMinimum(0.5)
        self.cross_section_top_spin.setMaximum(30.0)
        self.cross_section_top_spin.setSingleStep(0.5)
        self.cross_section_top_spin.setDecimals(1)
        self.cross_section_top_spin.setSuffix(" km")
        self.cross_section_top_spin.setValue(12.0)
        self.cross_section_top_spin.setToolTip("Top of vertical cross sections in km MSL")
        self.cross_section_top_spin.valueChanged.connect(self.render_current_time)
        form.addRow("X-sec top", self.cross_section_top_spin)

        self.state_borders_check = QCheckBox("Show state borders")
        self.state_borders_check.setChecked(True)
        self.state_borders_check.toggled.connect(self.render_current_time)
        form.addRow("Borders", self.state_borders_check)

        self.topography_check = QCheckBox("Show topography")
        self.topography_check.setChecked(False)
        self.topography_check.toggled.connect(self.render_current_time)
        form.addRow("Topo", self.topography_check)

        self.wind_barbs_check = QCheckBox("Show wind barbs")
        self.wind_barbs_check.setChecked(False)
        self.wind_barbs_check.toggled.connect(self.render_current_time)
        form.addRow("Wind", self.wind_barbs_check)

        self.wind_level_combo = QComboBox()
        for label, value in BARB_LEVEL_OPTIONS:
            self.wind_level_combo.addItem(label, value)
        self.wind_level_combo.currentIndexChanged.connect(self.render_current_time)
        form.addRow("Barb level", self.wind_level_combo)

        self.wind_barb_size_spin = QSpinBox()
        self.wind_barb_size_spin.setMinimum(1)
        self.wind_barb_size_spin.setMaximum(20)
        self.wind_barb_size_spin.setValue(10)
        self.wind_barb_size_spin.setToolTip("Relative size of map wind barbs")
        self.wind_barb_size_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Barb size", self.wind_barb_size_spin)

        self.wind_barb_density_spin = QSpinBox()
        self.wind_barb_density_spin.setMinimum(4)
        self.wind_barb_density_spin.setMaximum(40)
        self.wind_barb_density_spin.setValue(22)
        self.wind_barb_density_spin.setToolTip("Higher values draw denser map wind barbs")
        self.wind_barb_density_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Barb density", self.wind_barb_density_spin)

        self.hodograph_check = QCheckBox("Show Skew-T")
        self.hodograph_check.setChecked(False)
        self.hodograph_check.toggled.connect(self.render_current_time)
        form.addRow("Skew-T", self.hodograph_check)

        self.side_hodograph_check = QCheckBox("Embed hodograph")
        self.side_hodograph_check.setChecked(True)
        self.side_hodograph_check.toggled.connect(self.render_current_time)
        form.addRow("Side hodo", self.side_hodograph_check)

        self.hodograph_overlay_check = QCheckBox("Overlay hodograph")
        self.hodograph_overlay_check.setChecked(False)
        self.hodograph_overlay_check.toggled.connect(self.render_current_time)
        form.addRow("Hodograph", self.hodograph_overlay_check)

        self.hodo_size_spin = QSpinBox()
        self.hodo_size_spin.setMinimum(1)
        self.hodo_size_spin.setMaximum(20)
        self.hodo_size_spin.setValue(20)
        self.hodo_size_spin.setToolTip("Relative size of map hodograph overlays")
        self.hodo_size_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Hodo size", self.hodo_size_spin)

        self.hodo_spacing_spin = QSpinBox()
        self.hodo_spacing_spin.setMinimum(2)
        self.hodo_spacing_spin.setMaximum(20)
        self.hodo_spacing_spin.setValue(7)
        self.hodo_spacing_spin.setToolTip("Spacing of map hodograph overlays")
        self.hodo_spacing_spin.valueChanged.connect(self.render_current_time)
        form.addRow("Hodo spacing", self.hodo_spacing_spin)

        self.time_label = QLabel("No timesteps loaded")
        self.time_label.setWordWrap(True)
        form.addRow("Time", self.time_label)

        self.point_label = QLabel("Sounding point: domain center")
        self.point_label.setWordWrap(True)
        form.addRow("Point", self.point_label)

        self.cross_section_label = QLabel("Cross section: off")
        self.cross_section_label.setWordWrap(True)
        form.addRow("X-sec", self.cross_section_label)

        controls_layout.addLayout(form)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.on_time_changed)
        controls_layout.addWidget(self.time_slider)

        save_png_button = QPushButton("Save PNG")
        save_png_button.clicked.connect(self.save_png)
        controls_layout.addWidget(save_png_button)

        export_gif_button = QPushButton("Export GIF")
        export_gif_button.clicked.connect(self.export_gif)
        controls_layout.addWidget(export_gif_button)

        controls_layout.addStretch(1)

        self.figure = Figure(figsize=(12, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        root.addWidget(controls)
        root.addWidget(self.canvas)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        self.setCentralWidget(root)

    def _populate_variables(self):
        self.category_combo.blockSignals(True)
        self.variable_combo.blockSignals(True)

        grouped_variables = {category: [] for category in VARIABLE_CATEGORY_ORDER}
        for item in self.variable_metadata:
            category = categorize_variable(item)
            grouped_variables.setdefault(category, []).append(item)

        self.variables_by_category = {
            category: sorted(
                variables,
                key=lambda variable: (
                    DISPLAY_NAME_OVERRIDES.get(variable["name"], variable["description"]).lower(),
                    variable["name"].lower(),
                ),
            )
            for category, variables in grouped_variables.items()
            if variables
        }

        self.category_combo.clear()
        for category in VARIABLE_CATEGORY_ORDER:
            variables = self.variables_by_category.get(category, [])
            if not variables:
                continue
            self.category_combo.addItem(f"{category} ({len(variables)})", category)

        if self.category_combo.count():
            self.category_combo.setCurrentIndex(0)
        self._populate_variable_options()

        self.category_combo.blockSignals(False)
        self.variable_combo.blockSignals(False)
        self._sync_units_to_selection()

    def _populate_variable_options(self, preferred_variable_name=None):
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()

        category = self.selected_category()
        for item in self.variables_by_category.get(category, []):
            display_name = DISPLAY_NAME_OVERRIDES.get(item["name"], item["description"])
            label = f"{item['name']} - {display_name} [{display_units_name(item['units'])}]"
            self.variable_combo.addItem(label, item)

        if self.variable_combo.count():
            target_name = preferred_variable_name or self.variable_combo.itemData(0)["name"]
            index = next(
                (
                    combo_index
                    for combo_index in range(self.variable_combo.count())
                    if (self.variable_combo.itemData(combo_index) or {}).get("name") == target_name
                ),
                0,
            )
            self.variable_combo.setCurrentIndex(index)

        self.variable_combo.blockSignals(False)

    def _sync_units_to_selection(self):
        item = self.variable_combo.currentData()
        if not item:
            return
        self.units_edit.setText(display_units_name(item["units"]))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WRFOUT file",
            "",
            "WRF output files (*wrfout* *.nc *.cdf *.*)",
        )
        if path:
            self.load_file(path)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder With WRFOUT Files", "")
        if folder:
            self.load_folder(folder)

    def _reset_loaded_data(self):
        self._close_raw_datasets()
        self.wrf_file = None
        self.wrf_files = []
        self.raw_datasets = []
        self.file_path = None
        self.file_paths = []
        self.times = []
        self.time_sources = []
        self.lat = None
        self.lon = None
        self.selected_point = None
        self.cross_section_points = []
        self.map_axes = None
        self._style_range_cache = {}
        self._track_max_cache = {}
        self._update_point_label()
        self._update_cross_section_label()
        self._refresh_cross_section_window()

    def _close_raw_datasets(self):
        for dataset in self.raw_datasets:
            try:
                dataset.close()
            except Exception:
                pass

    def _set_loaded_state(self):
        self.wrf_file = self.wrf_files[0] if self.wrf_files else None
        self.file_path = self.file_paths[0] if self.file_paths else None
        self.file_label.setText(self.loaded_source_label())
        self.time_slider.setEnabled(bool(self.times))
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(max(len(self.times) - 1, 0))
        self._style_range_cache = {}
        self._track_max_cache = {}
        self.time_slider.setValue(0)
        self._sync_units_to_selection()
        self._update_point_label()
        self._update_cross_section_label()
        self.render_current_time()

    def _iter_candidate_files(self, folder):
        folder_path = Path(folder)
        patterns = ("wrfout*", "*.nc", "*.cdf")
        seen_paths = set()
        candidates = []
        for pattern in patterns:
            for path in sorted(folder_path.glob(pattern)):
                resolved = path.resolve()
                if not path.is_file() or resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                candidates.append(path)
        return candidates

    def _load_sources(self, paths):
        self._close_raw_datasets()
        loaded_files = []
        loaded_datasets = []
        loaded_paths = []
        loaded_times = []
        time_sources = []
        load_errors = []

        for path in paths:
            raw_dataset = None
            try:
                raw_dataset = Dataset(path)
                wrf_file = wrf.WrfFile(str(path))
                file_times = list(wrf_file.times())
            except Exception as exc:
                if raw_dataset is not None:
                    try:
                        raw_dataset.close()
                    except Exception:
                        pass
                load_errors.append(f"{Path(path).name}: {exc}")
                continue

            if not file_times:
                try:
                    raw_dataset.close()
                except Exception:
                    pass
                continue

            loaded_files.append(wrf_file)
            loaded_datasets.append(raw_dataset)
            loaded_paths.append(Path(path))
            file_index = len(loaded_files) - 1
            loaded_times.extend(file_times)
            time_sources.extend((file_index, timeidx) for timeidx in range(len(file_times)))

        if not loaded_files:
            for dataset in loaded_datasets:
                try:
                    dataset.close()
                except Exception:
                    pass
            if load_errors:
                raise RuntimeError(
                    "No readable WRFOUT files with timesteps were found.\n" + "\n".join(load_errors[:5])
                )
            raise RuntimeError("No readable WRFOUT files with timesteps were found.")

        self.wrf_files = loaded_files
        self.raw_datasets = loaded_datasets
        self.file_paths = loaded_paths
        self.times = loaded_times
        self.time_sources = time_sources
        self.lat = as_float_array(self.wrf_files[0].getvar("lat", timeidx=0))
        self.lon = as_float_array(self.wrf_files[0].getvar("lon", timeidx=0))
        self.selected_point = None
        self.cross_section_points = []
        self.map_axes = None

    def load_file(self, path):
        try:
            self._load_sources([Path(path)])
        except Exception as exc:
            self._reset_loaded_data()
            QMessageBox.critical(self, "Failed to open file", str(exc))
            return

        self._set_loaded_state()

    def load_folder(self, folder):
        try:
            candidate_paths = self._iter_candidate_files(folder)
            if not candidate_paths:
                raise RuntimeError("No matching WRFOUT files were found in that folder.")
            self._load_sources(candidate_paths)
        except Exception as exc:
            self._reset_loaded_data()
            QMessageBox.critical(self, "Failed to open folder", str(exc))
            return

        self._set_loaded_state()

    def loaded_source_label(self):
        if not self.file_paths:
            return "No file loaded"
        if len(self.file_paths) == 1:
            return str(self.file_paths[0])
        return f"{self.file_paths[0].parent} ({len(self.file_paths)} files, {len(self.times)} timesteps)"

    def _resolve_time_source_info(self, timeidx):
        if not self.time_sources:
            raise RuntimeError("No WRFOUT file is loaded.")
        safe_timeidx = min(max(timeidx, 0), len(self.time_sources) - 1)
        file_index, local_timeidx = self.time_sources[safe_timeidx]
        return file_index, local_timeidx

    def _resolve_time_source(self, timeidx):
        file_index, local_timeidx = self._resolve_time_source_info(timeidx)
        return self.wrf_files[file_index], local_timeidx

    def _resolve_raw_time_source(self, timeidx):
        file_index, local_timeidx = self._resolve_time_source_info(timeidx)
        return self.raw_datasets[file_index], local_timeidx

    def _extract_dataset_variable(self, dataset, raw_name, local_timeidx):
        variable = dataset.variables[raw_name]
        if variable.ndim >= 1 and variable.dimensions and variable.dimensions[0] == "Time":
            data = variable[local_timeidx, ...]
        else:
            data = variable[:]
        return as_float_array(data)

    def _get_track_source_field(self, source_name, timeidx, units=None):
        try:
            return as_float_array(self._getvar(source_name, timeidx, units=units))
        except Exception:
            if units is None:
                raise
            return as_float_array(self._getvar(source_name, timeidx, units=None))

    def _get_track_max_variable(self, variable_name, units=None):
        cache_key = (variable_name, units or "")
        cached = self._track_max_cache.get(cache_key)
        if cached is not None:
            return cached

        metadata = TRACK_MAX_VARIABLES[variable_name]
        source_names = metadata.get("source_names", [])
        if not source_names:
            raise RuntimeError(f"No source variables are configured for {variable_name}.")

        max_field = None
        source_used = None
        errors = []
        for timeidx in range(len(self.times)):
            source_field = None
            for source_name in source_names:
                try:
                    source_field = self._get_track_source_field(source_name, timeidx, units=units)
                    source_used = source_name
                    break
                except Exception as exc:
                    if timeidx == 0:
                        errors.append(f"{source_name}: {exc}")

            if source_field is None:
                continue
            if source_field.ndim >= 3:
                source_field = np.nanmax(source_field, axis=0)
            if source_field.ndim != 2:
                raise ValueError(
                    f"{variable_name} requires a 2D source field; {source_used} has shape {source_field.shape}."
                )

            if max_field is None:
                max_field = np.array(source_field, dtype=float, copy=True)
            else:
                if source_field.shape != max_field.shape:
                    raise ValueError(
                        f"{source_used} shape changed from {max_field.shape} to {source_field.shape}."
                    )
                max_field = np.fmax(max_field, source_field)

        if max_field is None:
            detail = "; ".join(errors) if errors else "no source data was available"
            raise RuntimeError(f"Unable to build {metadata['description']}: {detail}")

        self._track_max_cache[cache_key] = max_field
        return max_field

    def _getvar(self, variable_name, timeidx, units=None):
        if (
            variable_name in MIXING_RATIO_VARIABLES
            or variable_name in DIRECT_DATASET_VARIABLES
            or variable_name in PBL_VARIABLES
            or variable_name in TRACK_MAX_VARIABLES
            or variable_name == "planetary_albedo"
            or variable_name == "surface_precip"
        ):
            return self._get_custom_variable(variable_name, timeidx, units=units)
        wrf_file, local_timeidx = self._resolve_time_source(timeidx)
        kwargs = {"timeidx": local_timeidx}
        if units is not None:
            kwargs["units"] = units
        return wrf_file.getvar(variable_name, **kwargs)

    def _get_custom_variable(self, variable_name, timeidx, units=None):
        dataset, local_timeidx = self._resolve_raw_time_source(timeidx)
        requested_units = (units or "").strip().lower()

        if variable_name in TRACK_MAX_VARIABLES:
            return self._get_track_max_variable(variable_name, units=units)

        if variable_name in PBL_VARIABLES:
            metadata = PBL_VARIABLES[variable_name]
            for raw_name in metadata.get("fetch_names", [variable_name]):
                if raw_name in dataset.variables:
                    return self._extract_dataset_variable(dataset, raw_name, local_timeidx)
            tried_names = ", ".join(metadata.get("fetch_names", [variable_name]))
            raise RuntimeError(f"None of these PBL variables were found in the file: {tried_names}")

        if variable_name == "planetary_albedo":
            if "SWUPT" in dataset.variables and "SWDNT" in dataset.variables:
                reflected = as_float_array(dataset.variables["SWUPT"][local_timeidx, ...])
                incoming = as_float_array(dataset.variables["SWDNT"][local_timeidx, ...])
                ratio = np.divide(
                    reflected,
                    incoming,
                    out=np.full_like(reflected, np.nan, dtype=float),
                    where=np.isfinite(incoming) & (incoming > 1.0),
                )
                values = np.clip(ratio, 0.0, 1.0)
            else:
                swdown = as_float_array(dataset.variables["SWDOWN"][local_timeidx, ...])
                coszen = as_float_array(dataset.variables["COSZEN"][local_timeidx, ...])
                incoming_toa = 1361.0 * np.maximum(coszen, 0.0)
                transmitted = np.divide(
                    swdown,
                    incoming_toa,
                    out=np.full_like(swdown, np.nan, dtype=float),
                    where=np.isfinite(incoming_toa) & (incoming_toa > 1.0),
                )
                # Fallback approximation when TOA reflected flux is unavailable.
                values = np.clip(1.0 - transmitted, 0.0, 1.0)

            if requested_units in {"", "%", "percent", "percentage"}:
                return values * 100.0
            if requested_units in {"fraction", "frac", "0-1"}:
                return values
            raise ValueError(
                f"Unsupported units '{units}' for {variable_name}. Use % or fraction."
            )

        if variable_name in MIXING_RATIO_VARIABLES:
            raw_name = MIXING_RATIO_VARIABLES[variable_name]
            values = np.maximum(as_float_array(dataset.variables[raw_name][local_timeidx, ...]), 0.0)
            if requested_units in {"", "g/kg", "g kg-1", "g kg^-1"}:
                return values * 1000.0
            if requested_units in {"kg/kg", "kg kg-1", "kg kg^-1"}:
                return values
            raise ValueError(
                f"Unsupported units '{units}' for {variable_name}. Use g/kg or kg/kg."
            )

        if variable_name in DIRECT_DATASET_VARIABLES:
            values = as_float_array(dataset.variables[variable_name][local_timeidx, ...])
            if variable_name == "ALBEDO":
                finite_values = values[np.isfinite(values)]
                stored_as_fraction = bool(
                    finite_values.size and float(np.nanmax(finite_values)) <= 1.0
                )
                if requested_units in {"", "%", "percent", "percentage"}:
                    return values * 100.0 if stored_as_fraction else values
                if requested_units in {"fraction", "frac", "0-1"}:
                    return values if stored_as_fraction else values / 100.0
                raise ValueError(
                    f"Unsupported units '{units}' for {variable_name}. Use % or fraction."
                )
            if variable_name == "RTHRATSW":
                if requested_units in {"", "k/s", "k s-1", "k s^-1"}:
                    return values
                if requested_units in {"k/hr", "k h-1", "k h^-1"}:
                    return values * 3600.0
                raise ValueError(
                    f"Unsupported units '{units}' for {variable_name}. Use K/s or K/hr."
                )

            if requested_units in {"", "w/m^2", "w m-2", "w m^-2"}:
                return values
            if requested_units in {"kw/m^2", "kw m-2", "kw m^-2"}:
                return values / 1000.0
            raise ValueError(
                f"Unsupported units '{units}' for {variable_name}. Use W/m^2 or kW/m^2."
            )

        rainc = as_float_array(dataset.variables["RAINC"][local_timeidx, ...])
        rainnc = as_float_array(dataset.variables["RAINNC"][local_timeidx, ...])
        total_precip = np.maximum(rainc + rainnc, 0.0)
        if requested_units in {"", "mm"}:
            return total_precip
        if requested_units in {"cm"}:
            return total_precip / 10.0
        if requested_units in {"in", "inch", "inches"}:
            return total_precip / 25.4
        raise ValueError(
            f"Unsupported units '{units}' for surface_precip. Use mm, cm, or inches."
        )

    def selected_variable(self):
        item = self.variable_combo.currentData()
        return item["name"] if item else None

    def selected_category(self):
        return self.category_combo.currentData()

    def selected_units(self):
        text = self.units_edit.text().strip()
        return request_units_name(text or None)

    def selected_units_label(self):
        text = self.units_edit.text().strip()
        return text or None

    def selected_display_name(self):
        item = self.variable_combo.currentData() or {}
        variable_name = item.get("name")
        return DISPLAY_NAME_OVERRIDES.get(variable_name, variable_name or "")

    def _finite_limits(self, finite_values):
        if finite_values is None:
            return None
        values = np.asarray(finite_values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.nanmin(values)), float(np.nanmax(values))

    def _style_range_key(self, style_kind):
        return (
            style_kind,
            self.selected_variable() or "",
            (self.selected_units() or "").strip().lower(),
        )

    def _positive_step(self, levels, lower_edge=False):
        values = np.asarray(levels, dtype=float)
        diffs = np.diff(values)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 1.0
        return float(diffs[0] if lower_edge else diffs[-1])

    def _clamped_ticks(self, ticks, vmin, vmax):
        values = np.asarray(ticks, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return values

        lower = float(min(vmin, vmax))
        upper = float(max(vmin, vmax))
        epsilon = max((upper - lower) * 1e-10, 1e-12)
        values = values[(values >= lower - epsilon) & (values <= upper + epsilon)]
        values = np.clip(values, lower, upper)
        return np.unique(values)

    def _expand_discrete_table_to_limits(self, levels, colors, limits):
        levels = [float(value) for value in levels]
        colors = list(colors)
        if limits is None or len(levels) < 2 or not colors:
            return levels, colors

        data_min, data_max = limits
        lower_step = self._positive_step(levels, lower_edge=True)
        upper_step = self._positive_step(levels, lower_edge=False)

        if data_min < levels[0]:
            lower_count = int(np.ceil((levels[0] - data_min) / lower_step))
            lower_levels = [
                levels[0] - lower_step * offset
                for offset in range(lower_count, 0, -1)
            ]
            levels = lower_levels + levels
            colors = [colors[0]] * lower_count + colors

        if data_max > levels[-1]:
            upper_count = int(np.ceil((data_max - levels[-1]) / upper_step))
            upper_levels = [
                levels[-1] + upper_step * offset
                for offset in range(1, upper_count + 1)
            ]
            levels = levels + upper_levels
            colors = colors + [colors[-1]] * upper_count

        return levels, colors

    def _fixed_discrete_table(self, levels, colors, finite_values):
        base_levels = [float(value) for value in levels]
        base_colors = list(colors)
        signature = (
            len(base_levels),
            base_levels[0] if base_levels else None,
            base_levels[-1] if base_levels else None,
            len(base_colors),
        )
        key = self._style_range_key("discrete")
        cached = self._style_range_cache.get(key)
        if cached is None or cached.get("signature") != signature:
            cached_levels = base_levels
            cached_colors = base_colors
        else:
            cached_levels = cached["levels"]
            cached_colors = cached["colors"]

        levels, colors = self._expand_discrete_table_to_limits(
            cached_levels,
            cached_colors,
            self._finite_limits(finite_values),
        )
        self._style_range_cache[key] = {
            "signature": signature,
            "levels": levels,
            "colors": colors,
        }
        return levels, colors

    def _continuous_limits(self, finite_values):
        limits = self._finite_limits(finite_values)
        key = self._style_range_key("continuous")
        cached = self._style_range_cache.get(key)

        if limits is None:
            if cached is None:
                return None
            return cached["vmin"], cached["vmax"], cached["ticks"]

        data_min, data_max = limits
        if cached is None:
            if data_min == data_max:
                pad = max(abs(data_min) * 0.05, 1.0)
                tick_values = MaxNLocator(nbins=12).tick_values(data_min - pad, data_max + pad)
            else:
                tick_values = MaxNLocator(nbins=12).tick_values(data_min, data_max)
            vmin = float(tick_values[0])
            vmax = float(tick_values[-1])
            step = self._positive_step(tick_values, lower_edge=False)
        else:
            vmin = float(cached["vmin"])
            vmax = float(cached["vmax"])
            step = float(cached["step"])

        if step <= 0 or not np.isfinite(step):
            step = max((vmax - vmin) / 12.0, abs(data_max) * 0.05, 1.0)

        if data_min < vmin:
            vmin -= np.ceil((vmin - data_min) / step) * step
        if data_max > vmax:
            vmax += np.ceil((data_max - vmax) / step) * step
        if vmin == vmax:
            pad = max(abs(vmin) * 0.05, 1.0)
            vmin -= pad
            vmax += pad

        ticks = self._clamped_ticks(MaxNLocator(nbins=12).tick_values(vmin, vmax), vmin, vmax)
        self._style_range_cache[key] = {
            "vmin": float(vmin),
            "vmax": float(vmax),
            "step": float(step),
            "ticks": ticks,
        }
        return float(vmin), float(vmax), ticks

    def _discrete_ticks(self, levels, max_ticks=12):
        if len(levels) <= max_ticks:
            return levels
        locator = MaxNLocator(nbins=max_ticks)
        return self._clamped_ticks(
            locator.tick_values(float(levels[0]), float(levels[-1])),
            float(levels[0]),
            float(levels[-1]),
        )

    def _make_discrete_style(self, levels, colors, extend="both", ticks=None, finite_values=None):
        base_levels = [float(value) for value in levels]
        levels, colors = self._fixed_discrete_table(base_levels, colors, finite_values)
        cmap = ListedColormap(colors)
        cmap = cmap.copy()
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
        cmap.set_bad(colors[0])
        norm = BoundaryNorm(levels, cmap.N)
        use_custom_ticks = (
            ticks is not None
            and len(levels) == len(base_levels)
            and levels[0] == base_levels[0]
            and levels[-1] == base_levels[-1]
        )
        return {
            "cmap": cmap,
            "norm": norm,
            "levels": levels,
            "boundaries": levels,
            "extend": extend,
            "ticks": ticks if use_custom_ticks else self._discrete_ticks(levels),
        }

    def _make_continuous_style(self, finite_values=None):
        cmap = cm.get_cmap("turbo").copy()
        cmap.set_under(cmap(0.0))
        cmap.set_over(cmap(1.0))
        cmap.set_bad(cmap(0.0))
        limits = self._continuous_limits(finite_values)
        if limits is None:
            return {
                "cmap": cmap,
                "norm": None,
                "levels": None,
                "boundaries": None,
                "extend": None,
                "ticks": None,
                "continuous": True,
            }

        vmin, vmax, ticks = limits
        return {
            "cmap": cmap,
            "norm": Normalize(vmin=vmin, vmax=vmax),
            "levels": None,
            "boundaries": None,
            "extend": "neither",
            "ticks": ticks,
            "continuous": True,
        }

    def _add_field_colorbar(self, figure, mesh, style, units, axes=None, colorbar_spec=None, **kwargs):
        mappable = mesh
        colorbar_kwargs = dict(kwargs)
        if style.get("continuous") and style.get("norm") is not None:
            mappable = cm.ScalarMappable(norm=style["norm"], cmap=style["cmap"])
            mappable.set_clim(float(style["norm"].vmin), float(style["norm"].vmax))
            mappable.set_array([])
        elif style.get("boundaries") is not None:
            colorbar_kwargs.setdefault("boundaries", style["boundaries"])
            colorbar_kwargs.setdefault("spacing", "proportional")

        if colorbar_spec is not None:
            colorbar_axes = figure.add_subplot(colorbar_spec)
            colorbar = figure.colorbar(
                mappable,
                cax=colorbar_axes,
                extend=style["extend"] or "neither",
                **colorbar_kwargs,
            )
        else:
            colorbar = figure.colorbar(
                mappable,
                ax=axes,
                extend=style["extend"] or "neither",
                **colorbar_kwargs,
            )
        if style["ticks"] is not None:
            ticks = style["ticks"]
            if style.get("norm") is not None:
                ticks = self._clamped_ticks(ticks, float(style["norm"].vmin), float(style["norm"].vmax))
            elif style.get("levels") is not None:
                ticks = self._clamped_ticks(ticks, float(style["levels"][0]), float(style["levels"][-1]))
            colorbar.set_ticks(ticks)
        if style.get("continuous") and style.get("norm") is not None:
            colorbar.ax.set_ylim(float(style["norm"].vmin), float(style["norm"].vmax))
        if units:
            colorbar.set_label(units)
        return colorbar

    def _field_style(self, finite_values=None):
        variable = self.selected_variable()
        units_name = (self.selected_units() or "").lower()

        if variable in {"dbz", "maxdbz", "maxdbz_uhel_fill"}:
            try:
                table = load_autumnplot_colortable(AUTUMNPLOT_JSON_FILES[variable])
            except (KeyError, OSError, ValueError):
                return self._make_discrete_style(
                    PIVOTAL_DBZ_LEVELS,
                    PIVOTAL_DBZ_COLORS,
                    extend="both",
                    finite_values=finite_values,
                )
            return self._make_discrete_style(
                table["levels"],
                table["colors"],
                extend="both",
                finite_values=finite_values,
            )

        if variable in {"td", "dp2m"}:
            try:
                table = load_autumnplot_colortable(AUTUMNPLOT_JSON_FILES[variable])
            except (KeyError, OSError, ValueError):
                if "f" in units_name:
                    levels = [value * 9.0 / 5.0 + 32.0 for value in PIVOTAL_DEWPOINT_LEVELS_C]
                elif units_name in {"k", "kelvin"}:
                    levels = [value + 273.15 for value in PIVOTAL_DEWPOINT_LEVELS_C]
                else:
                    levels = PIVOTAL_DEWPOINT_LEVELS_C
                return self._make_discrete_style(
                    levels,
                    PIVOTAL_DEWPOINT_COLORS,
                    extend="both",
                    finite_values=finite_values,
                )
            levels = convert_autumnplot_levels(table["levels"], units_name)
            return self._make_discrete_style(
                levels,
                table["colors"],
                extend="both",
                finite_values=finite_values,
            )

        if variable in {"tc", "t2", "temp"}:
            try:
                table = load_autumnplot_colortable(AUTUMNPLOT_JSON_FILES[variable])
            except (KeyError, OSError, ValueError):
                if "f" in units_name:
                    levels = PIVOTAL_TEMP_LEVELS_F
                elif units_name in {"k", "kelvin"}:
                    levels = f_to_k(PIVOTAL_TEMP_LEVELS_F)
                else:
                    levels = f_to_c(PIVOTAL_TEMP_LEVELS_F)
                return self._make_discrete_style(
                    levels,
                    PIVOTAL_TEMP_COLORS,
                    extend="both",
                    finite_values=finite_values,
                )
            levels = convert_autumnplot_levels(table["levels"], units_name)
            return self._make_discrete_style(
                levels,
                table["colors"],
                extend="both",
                finite_values=finite_values,
            )

        if variable in DERIVED_ALBEDO_VARIABLES:
            if units_name in {"fraction", "frac", "0-1"}:
                levels = np.linspace(0.0, 1.0, 11)
                ticks = levels
            else:
                levels = np.arange(0.0, 110.0, 10.0)
                ticks = levels
            colors = [
                "#1f4e79",
                "#2f6fab",
                "#4f96c6",
                "#7dc2d2",
                "#b7e0d0",
                "#f2e8b6",
                "#f3cf7a",
                "#eba45a",
                "#d96d43",
                "#b33c2e",
            ]
            return self._make_discrete_style(
                levels,
                colors,
                extend="neither",
                ticks=ticks,
                finite_values=finite_values,
            )

        return self._make_continuous_style(finite_values)

    def _use_old_dbz_rendering(self):
        return self.selected_variable() in {"dbz", "maxdbz", "maxdbz_uhel_fill"}

    def _is_track_max_variable(self):
        return self.selected_variable() in TRACK_MAX_VARIABLES

    def _field_time_label(self, timeidx):
        if self._is_track_max_variable():
            return "all loaded times"
        return self.times[timeidx] if self.times else "time 0"

    def domain_label(self):
        if not self.file_path:
            return "Domain: N/A"

        domain = "unknown"
        parts = self.file_path.name.split("_")
        if len(parts) > 1 and parts[1].lower().startswith("d"):
            domain = parts[1]

        if self.wrf_file:
            dx_km = getattr(self.wrf_file, "dx", None)
            dy_km = getattr(self.wrf_file, "dy", None)
            nx = getattr(self.wrf_file, "nx", None)
            ny = getattr(self.wrf_file, "ny", None)
            resolution = ""
            if dx_km is not None and dy_km is not None:
                resolution = f" | {dx_km / 1000:.1f} x {dy_km / 1000:.1f} km"
            size = ""
            if nx is not None and ny is not None:
                size = f" | {nx} x {ny}"
            count = f" | {len(self.file_paths)} files" if len(self.file_paths) > 1 else ""
            return f"Domain {domain}{resolution}{size}{count}"

        return f"Domain {domain}"

    def _default_point(self):
        if self.lat is None or self.lon is None:
            return None
        return (self.lat.shape[0] // 2, self.lat.shape[1] // 2)

    def _active_point(self):
        return self.selected_point or self._default_point()

    def _format_grid_point(self, point):
        if point is None or self.lat is None or self.lon is None:
            return "unavailable"

        j, i = point
        lat = float(self.lat[j, i])
        lon = float(self.lon[j, i])
        return f"{lat:.2f}, {lon:.2f} (j={j}, i={i})"

    def _update_point_label(self):
        point = self._active_point()
        if point is None or self.lat is None or self.lon is None:
            self.point_label.setText("Sounding point: unavailable")
            return

        j, i = point
        lat = float(self.lat[j, i])
        lon = float(self.lon[j, i])
        source = "selected" if self.selected_point else "domain center"
        self.point_label.setText(f"{source}: {lat:.2f}, {lon:.2f} (j={j}, i={i})")

    def _update_cross_section_label(self):
        if self.lat is None or self.lon is None:
            self.cross_section_label.setText("Cross section: unavailable")
            return

        enabled = self.cross_section_button.isChecked()
        if not enabled:
            self.cross_section_label.setText("Cross section: off")
            return

        if not self.cross_section_points:
            self.cross_section_label.setText("on: click a start point on the map")
            return

        if len(self.cross_section_points) == 1:
            start_text = self._format_grid_point(self.cross_section_points[0])
            self.cross_section_label.setText(f"on: start {start_text}; click the end point")
            return

        start_text = self._format_grid_point(self.cross_section_points[0])
        end_text = self._format_grid_point(self.cross_section_points[1])
        self.cross_section_label.setText(f"on: {start_text} -> {end_text}")

    def _ensure_cross_section_window(self):
        if self.cross_section_window is None:
            self.cross_section_window = CrossSectionWindow(self)
            self.cross_section_window.closed_callback = self._on_cross_section_window_closed
        return self.cross_section_window

    def _on_cross_section_window_closed(self):
        if not self.cross_section_button.isChecked():
            return
        self.cross_section_button.blockSignals(True)
        self.cross_section_button.setChecked(False)
        self.cross_section_button.blockSignals(False)
        self._update_cross_section_label()
        if self.wrf_files:
            self.render_current_time()
        self.statusBar().showMessage("Vertical cross section tool disabled.")

    def on_cross_section_toggled(self, checked):
        self._update_cross_section_label()
        if checked:
            window = self._ensure_cross_section_window()
            window.show()
            window.raise_()
            window.activateWindow()
            message = (
                "Vertical cross section mode enabled. Click the start point on the map."
                if not self.cross_section_points
                else (
                    "Start point selected. Click the end point on the map."
                    if len(self.cross_section_points) == 1
                    else "Vertical cross section mode enabled."
                )
            )
        else:
            if self.cross_section_window is not None:
                self.cross_section_window.hide()
            message = "Vertical cross section tool disabled."

        if self.wrf_files:
            self.render_current_time()
        else:
            self._refresh_cross_section_window()
        self.statusBar().showMessage(message)

    def _nearest_grid_point(self, lat_value, lon_value):
        if self.lat is None or self.lon is None:
            return None

        distance = (self.lon - lon_value) ** 2 + (self.lat - lat_value) ** 2
        if not np.isfinite(distance).any():
            return None

        j, i = np.unravel_index(np.nanargmin(distance), distance.shape)
        return int(j), int(i)

    def _handle_cross_section_click(self, point):
        if len(self.cross_section_points) >= 2:
            self.cross_section_points = [point]
            self._update_cross_section_label()
            return f"Cross-section start point reset to {self._format_grid_point(point)}"

        self.cross_section_points.append(point)
        self._update_cross_section_label()
        if len(self.cross_section_points) == 1:
            return f"Selected cross-section start point at {self._format_grid_point(point)}"
        return (
            f"Selected cross-section end point at {self._format_grid_point(point)}. "
            "Generating vertical cross section."
        )

    def on_variable_changed(self):
        self._sync_units_to_selection()
        self.render_current_time()

    def on_category_changed(self):
        previous_variable = self.selected_variable()
        self._populate_variable_options(preferred_variable_name=previous_variable)
        self._sync_units_to_selection()
        self.render_current_time()

    def on_time_changed(self):
        value = self.time_slider.value()
        if self.times:
            self.time_label.setText(f"{value + 1}/{len(self.times)}: {self.times[value]}")
        self.render_current_time()

    def on_canvas_click(self, event):
        if self.lat is None or self.lon is None or event.xdata is None or event.ydata is None:
            return
        if self.map_axes is not None and event.inaxes is not self.map_axes:
            return

        point = self._nearest_grid_point(event.ydata, event.xdata)
        if point is None:
            return

        if self.cross_section_button.isChecked():
            message = self._handle_cross_section_click(point)
            self.render_current_time()
            self.statusBar().showMessage(message)
            return

        self.selected_point = point
        self._update_point_label()
        self.render_current_time()
        self.statusBar().showMessage(f"Selected sounding point at {self._format_grid_point(point)}")

    def _current_timeidx(self):
        return self.time_slider.value() if self.times else 0

    def _fetch_field(self, timeidx):
        if not self.wrf_files:
            raise RuntimeError("No WRFOUT file is loaded.")
        if self.selected_variable() == "maxdbz_uhel_fill":
            composite_reflectivity, _ = self._fetch_composite_reflectivity_uhel_fields(timeidx)
            return composite_reflectivity, 2, None, None
        data = self._getvar(
            self.selected_variable(),
            timeidx,
            units=self.selected_units(),
        )
        array = as_float_array(data)
        if self.selected_variable() == "dbz" and array.ndim >= 3:
            return np.nanmax(array, axis=0), 2, None, None
        if array.ndim == 3:
            level = min(self.level_spin.value(), array.shape[0] - 1)
            return array[level], array.ndim, level, array.shape[0]
        if array.ndim == 2:
            return array, array.ndim, None, None
        raise ValueError(f"Unsupported variable shape: {array.shape}")

    def _smoothed_field(self, field):
        sigma = self.smoothing_spin.value()
        if sigma <= 0:
            return field

        valid_mask = np.isfinite(field)
        if not valid_mask.any():
            return field

        filled = np.where(valid_mask, field, 0.0)
        smoothed_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
        smoothed_weights = gaussian_filter(valid_mask.astype(float), sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = smoothed_values / smoothed_weights
        smoothed[smoothed_weights == 0] = np.nan
        return smoothed

    def _cross_section_levels(self, style, finite_values):
        if style["levels"] is not None:
            return style["levels"]
        if style["norm"] is not None:
            return np.linspace(float(style["norm"].vmin), float(style["norm"].vmax), 64)
        return np.linspace(float(finite_values.min()), float(finite_values.max()), 64)

    def _surface_ribbon_grid(self, distance_km, terrain_profile, surface_values):
        distance = np.asarray(distance_km, dtype=float)
        terrain = np.asarray(terrain_profile, dtype=float)
        values = np.asarray(surface_values, dtype=float)

        axis_mask = np.isfinite(distance) & np.isfinite(terrain)
        if axis_mask.sum() < 2:
            raise RuntimeError("The cross-section path was too short to draw a smooth ribbon.")

        axis_distance = distance[axis_mask]
        axis_terrain = terrain[axis_mask]
        order = np.argsort(axis_distance)
        axis_distance = axis_distance[order]
        axis_terrain = axis_terrain[order]
        axis_distance, unique_indices = np.unique(axis_distance, return_index=True)
        axis_terrain = axis_terrain[unique_indices]

        value_mask = np.isfinite(distance) & np.isfinite(values)
        if value_mask.sum() == 0:
            raise RuntimeError("The selected line did not contain any valid field values.")

        value_distance = distance[value_mask]
        value_values = values[value_mask]
        order = np.argsort(value_distance)
        value_distance = value_distance[order]
        value_values = value_values[order]
        value_distance, unique_indices = np.unique(value_distance, return_index=True)
        value_values = value_values[unique_indices]

        sample_count = int(min(max(axis_distance.size * 8, 180), 2400))
        dense_distance = np.linspace(float(axis_distance[0]), float(axis_distance[-1]), sample_count)
        dense_terrain = np.interp(dense_distance, axis_distance, axis_terrain)
        if value_distance.size == 1:
            dense_values = np.full_like(dense_distance, float(value_values[0]))
        else:
            dense_values = np.interp(dense_distance, value_distance, value_values)

        ribbon_offsets = np.linspace(0.03, 0.16, 18)
        distance_grid = np.broadcast_to(dense_distance[np.newaxis, :], (ribbon_offsets.size, dense_distance.size))
        height_grid = dense_terrain[np.newaxis, :] + ribbon_offsets[:, np.newaxis]
        value_grid = np.broadcast_to(dense_values[np.newaxis, :], height_grid.shape)
        return distance_grid, height_grid, value_grid

    def _cross_section_top_km(self, y_bottom_km, max_terrain_km):
        requested_top_km = float(self.cross_section_top_spin.value())
        minimum_top_km = max(float(y_bottom_km) + 0.5, float(max_terrain_km) + 0.25)
        return max(requested_top_km, minimum_top_km)

    def _fetch_composite_reflectivity_uhel_fields(self, timeidx):
        composite_reflectivity = as_float_array(
            self._getvar("maxdbz", timeidx, units=self.selected_units())
        )
        uhel = as_float_array(self._getvar("uhel", timeidx))

        if composite_reflectivity.ndim >= 3:
            composite_reflectivity = np.nanmax(composite_reflectivity, axis=0)
        if uhel.ndim >= 3:
            uhel = np.nanmax(uhel, axis=0)

        if composite_reflectivity.ndim != 2 or uhel.ndim != 2:
            raise ValueError(
                "Composite Reflectivity & 1-hr UH overlay requires 2D maxdbz and uhel fields."
            )

        return composite_reflectivity, uhel

    def _cross_section_line_indices(self):
        if len(self.cross_section_points) < 2:
            raise RuntimeError("Select both a start point and an end point on the map.")

        start_point = self.cross_section_points[0]
        end_point = self.cross_section_points[1]
        if start_point == end_point:
            raise RuntimeError("Choose two different map points for the vertical cross section.")

        return _bresenham_indices(start_point[0], start_point[1], end_point[0], end_point[1])

    def _cross_section_source_candidates(self):
        candidates = []
        selected_variable = self.selected_variable()
        override_variable = CROSS_SECTION_SOURCE_OVERRIDES.get(selected_variable)
        if override_variable:
            candidates.append(override_variable)
        candidates.append(selected_variable)

        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate in seen or not candidate:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered

    def _resolve_cross_section_source(self, timeidx, model_level_count):
        units_name = self.selected_units()
        selected_variable = self.selected_variable()
        surface_field = None
        source_variable = None
        problems = []

        for candidate in self._cross_section_source_candidates():
            try:
                data = as_float_array(self._getvar(candidate, timeidx, units=units_name))
            except Exception as exc:
                problems.append(f"{candidate}: {exc}")
                continue

            if data.ndim == 3 and data.shape[0] == model_level_count and data.shape[-2:] == self.lat.shape:
                return {"kind": "volume", "data": data, "source_variable": candidate}

            if data.ndim == 2 and data.shape == self.lat.shape and surface_field is None:
                surface_field = data
                source_variable = candidate
                continue

            problems.append(f"{candidate}: incompatible shape {data.shape}")

        if surface_field is not None:
            return {"kind": "surface", "data": surface_field, "source_variable": source_variable}

        detail = problems[0] if problems else "no compatible 3D field was available"
        raise RuntimeError(
            f"Vertical cross section is unavailable for '{selected_variable}'. {detail}."
        )

    def _build_cross_section(self, timeidx):
        if self.lat is None or self.lon is None:
            raise RuntimeError("No WRF domain is loaded.")

        jj, ii = self._cross_section_line_indices()
        if jj.size < 2:
            raise RuntimeError("Choose two different map points for the vertical cross section.")

        height_msl = as_float_array(self._getvar("height", timeidx, units="m"))
        terrain = as_float_array(self._getvar("terrain", timeidx, units="m"))
        if height_msl.ndim != 3:
            raise RuntimeError("The current dataset does not provide 3D model heights.")
        if terrain.shape != self.lat.shape:
            raise RuntimeError("Terrain data does not match the loaded map grid.")

        source = self._resolve_cross_section_source(timeidx, height_msl.shape[0])
        distance_km = np.zeros(jj.size, dtype=float)
        for idx in range(1, jj.size):
            previous_j = jj[idx - 1]
            previous_i = ii[idx - 1]
            current_j = jj[idx]
            current_i = ii[idx]
            distance_km[idx] = distance_km[idx - 1] + _haversine_km(
                float(self.lat[previous_j, previous_i]),
                float(self.lon[previous_j, previous_i]),
                float(self.lat[current_j, current_i]),
                float(self.lon[current_j, current_i]),
            )

        terrain_km = terrain[jj, ii] / 1000.0
        finite_terrain = terrain_km[np.isfinite(terrain_km)]
        min_terrain_km = float(np.nanmin(finite_terrain)) if finite_terrain.size else 0.0
        max_terrain_km = float(np.nanmax(finite_terrain)) if finite_terrain.size else 0.0
        y_bottom_km = max(0.0, min_terrain_km - 0.25)
        y_top_km = self._cross_section_top_km(y_bottom_km, max_terrain_km)

        section = {
            "kind": source["kind"],
            "distance_km": distance_km,
            "terrain_km": terrain_km,
            "y_bottom_km": y_bottom_km,
            "y_top_km": y_top_km,
            "display_name": self.selected_display_name(),
            "selected_variable": self.selected_variable(),
            "source_variable": source["source_variable"],
            "time_label": self._field_time_label(timeidx),
            "start_point": self.cross_section_points[0],
            "end_point": self.cross_section_points[1],
            "units_label": self.selected_units_label()
            or display_units_name((self.variable_combo.currentData() or {}).get("units", "")),
        }

        if source["kind"] == "volume":
            field_section = source["data"][:, jj, ii]
            height_section_km = height_msl[:, jj, ii] / 1000.0

            reference_profile = height_section_km[:, 0]
            finite_reference = reference_profile[np.isfinite(reference_profile)]
            if finite_reference.size >= 2 and float(np.nanmean(np.diff(finite_reference))) < 0.0:
                field_section = field_section[::-1, :]
                height_section_km = height_section_km[::-1, :]

            finite_height = height_section_km[np.isfinite(height_section_km)]
            if finite_height.size == 0:
                raise RuntimeError("No valid height coordinates were available for this cross section.")

            section["field"] = field_section
            section["height_km"] = height_section_km
            section["model_top_km"] = float(np.nanmax(finite_height))
            return section

        section["surface_values"] = source["data"][jj, ii]
        return section

    def _render_cross_section_plot(self, window, section):
        figure = window.figure
        figure.clear()
        axes = figure.subplots()

        units_label = section["units_label"]
        info_lines = [
            f"{section['display_name']} | {section['time_label']}",
            f"Start: {self._format_grid_point(section['start_point'])}",
            f"End: {self._format_grid_point(section['end_point'])}",
            "Height axis: km MSL",
            f"Top: {section['y_top_km']:.1f} km MSL",
        ]
        if section["source_variable"] != section["selected_variable"]:
            info_lines.append(f"Cross-section source field: {section['source_variable']}")
        if section["kind"] == "surface":
            info_lines.append("Selected field has no vertical levels; shown as a terrain-following ribbon.")
        window.info_label.setText(" | ".join(info_lines))
        window.setWindowTitle(f"Vertical Cross Section - {section['display_name']}")

        terrain_profile = np.where(
            np.isfinite(section["terrain_km"]),
            section["terrain_km"],
            section["y_bottom_km"],
        )

        if section["kind"] == "volume":
            field = self._smoothed_field(section["field"])
            visible_mask = (
                np.isfinite(field)
                & np.isfinite(section["height_km"])
                & (section["height_km"] >= section["y_bottom_km"])
                & (section["height_km"] <= section["y_top_km"])
            )
            finite_values = field[visible_mask]
            if finite_values.size == 0:
                finite_values = field[np.isfinite(field)]
            if finite_values.size == 0:
                raise RuntimeError("The selected line did not contain any valid field values.")
            style = self._field_style(finite_values)

            distance_grid = np.broadcast_to(
                section["distance_km"][np.newaxis, :],
                section["height_km"].shape,
            )
            if style["levels"] is None:
                mesh = axes.pcolormesh(
                    distance_grid,
                    section["height_km"],
                    field,
                    cmap=style["cmap"],
                    norm=style["norm"],
                    shading="gouraud",
                )
            else:
                levels = self._cross_section_levels(style, finite_values)
                mesh = axes.contourf(
                    distance_grid,
                    section["height_km"],
                    field,
                    levels=levels,
                    cmap=style["cmap"],
                    norm=style["norm"],
                    antialiased=False,
                    extend=style["extend"] or "neither",
                )
        else:
            surface_values = self._smoothed_field(section["surface_values"])
            finite_values = surface_values[np.isfinite(surface_values)]
            if finite_values.size == 0:
                raise RuntimeError("The selected line did not contain any valid field values.")
            style = self._field_style(finite_values)

            distance_grid, ribbon_height, ribbon_values = self._surface_ribbon_grid(
                section["distance_km"],
                terrain_profile,
                surface_values,
            )
            if style["levels"] is None or np.unique(finite_values).size < 2:
                mesh = axes.pcolormesh(
                    distance_grid,
                    ribbon_height,
                    ribbon_values,
                    cmap=style["cmap"],
                    norm=style["norm"],
                    shading="gouraud",
                    zorder=5,
                )
            else:
                levels = self._cross_section_levels(style, finite_values)
                mesh = axes.contourf(
                    distance_grid,
                    ribbon_height,
                    ribbon_values,
                    levels=levels,
                    cmap=style["cmap"],
                    norm=style["norm"],
                    antialiased=False,
                    extend=style["extend"] or "neither",
                    zorder=5,
                )

        axes.fill_between(
            section["distance_km"],
            section["y_bottom_km"],
            terrain_profile,
            color="#7b5b3d",
            alpha=0.9,
            zorder=3,
        )
        axes.plot(
            section["distance_km"],
            terrain_profile,
            color="#2d1e10",
            linewidth=1.2,
            zorder=6,
        )
        axes.set_xlabel("Distance Along Section (km)")
        axes.set_ylabel("Height MSL (km)")
        axes.set_ylim(section["y_bottom_km"], section["y_top_km"])
        axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.28)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=9))
        axes.xaxis.set_major_locator(MaxNLocator(nbins=9))

        title = f"{section['display_name']} | {section['time_label']}"
        if section["kind"] == "surface":
            title += "\nSurface-only field (no vertical levels)"
        elif section["source_variable"] != section["selected_variable"]:
            title += f"\nSource field: {section['source_variable']}"
        if self.smoothing_spin.value() > 0:
            title += f" | smooth {self.smoothing_spin.value()}"
        axes.set_title(title)

        self._add_field_colorbar(
            figure,
            mesh,
            style,
            units_label,
            axes=axes,
            pad=0.02,
            shrink=0.9,
        )

        figure.tight_layout()
        window.canvas.draw_idle()

    def _refresh_cross_section_window(self):
        if self.cross_section_window is None:
            return
        if not self.cross_section_button.isChecked():
            self.cross_section_window.hide()
            return

        self.cross_section_window.show()
        if not self.wrf_files:
            self.cross_section_window.show_message(
                "Open a WRFOUT file, then click start and end points on the map."
            )
            return
        if not self.cross_section_points:
            self.cross_section_window.show_message(
                "Cross-section mode is on. Click the start point on the map."
            )
            return
        if len(self.cross_section_points) == 1:
            self.cross_section_window.show_message(
                f"Start point: {self._format_grid_point(self.cross_section_points[0])}\n"
                "Click the end point on the map."
            )
            return

        try:
            section = self._build_cross_section(self._current_timeidx())
            self._render_cross_section_plot(self.cross_section_window, section)
        except Exception as exc:
            self.cross_section_window.show_message(str(exc))

    def _wind_components_for_level(self, timeidx):
        level = self.wind_level_combo.currentData()
        if level == "surface":
            uv10 = as_float_array(self._getvar("uvmet10", timeidx, units="knots"))
            return uv10[0], uv10[1], "Surface (10 m)"

        pressure = as_float_array(self._getvar("pressure", timeidx, units="hPa"))
        uvmet = as_float_array(self._getvar("uvmet", timeidx, units="knots"))
        u_level = as_float_array(wrf.interplevel(uvmet[0], pressure, float(level)))
        v_level = as_float_array(wrf.interplevel(uvmet[1], pressure, float(level)))
        return u_level, v_level, f"{level} mb"

    def _draw_wind_barbs(self, axes, timeidx):
        if not self.wind_barbs_check.isChecked():
            return
        if self.lat is None or self.lon is None:
            return

        u_wind, v_wind, _ = self._wind_components_for_level(timeidx)
        density = max(1, self.wind_barb_density_spin.value())
        size_scale = self.wind_barb_size_spin.value() / 10.0
        skip = max(1, int(np.ceil(min(self.lat.shape) / density)))
        slc = (slice(None, None, skip), slice(None, None, skip))
        axes.barbs(
            self.lon[slc],
            self.lat[slc],
            u_wind[slc],
            v_wind[slc],
            length=3.8 * size_scale,
            linewidth=max(0.25, 0.4 * size_scale),
            sizes={
                "emptybarb": 0.07 * size_scale,
                "spacing": 0.24 * size_scale,
                "height": 0.3 * size_scale,
            },
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    def _draw_selected_point(self, axes):
        point = self._active_point()
        if point is None or self.lat is None or self.lon is None:
            return
        j, i = point
        axes.plot(
            float(self.lon[j, i]),
            float(self.lat[j, i]),
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    def _draw_cross_section_selection(self, axes):
        if not self.cross_section_button.isChecked():
            return
        if self.lat is None or self.lon is None or not self.cross_section_points:
            return

        line_lons = [float(self.lon[j, i]) for j, i in self.cross_section_points[:2]]
        line_lats = [float(self.lat[j, i]) for j, i in self.cross_section_points[:2]]
        if len(line_lons) == 2:
            axes.plot(
                line_lons,
                line_lats,
                color="#facc15",
                linewidth=2.2,
                linestyle="-",
                transform=ccrs.PlateCarree(),
                zorder=7,
            )

        markers = [("S", "#10b981"), ("E", "#ef4444")]
        for idx, point in enumerate(self.cross_section_points[:2]):
            j, i = point
            lon = float(self.lon[j, i])
            lat = float(self.lat[j, i])
            label, color = markers[idx]
            axes.plot(
                lon,
                lat,
                marker="o" if idx == 0 else "X",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
            axes.text(
                lon,
                lat,
                f" {label}",
                color=color,
                fontsize=9,
                fontweight="bold",
                transform=ccrs.PlateCarree(),
                zorder=9,
            )

    def _render_hodograph_overlay(self, parent_axes, timeidx):
        if not self.hodograph_overlay_check.isChecked():
            return

        if self.lat is None or self.lon is None or not self.wrf_files:
            return

        try:
            pressure = as_float_array(self._getvar("pressure", timeidx, units="hPa"))
            height_agl = as_float_array(self._getvar("height_agl", timeidx, units="m"))
            uvmet = as_float_array(self._getvar("uvmet", timeidx, units="knots"))
            uv10 = as_float_array(self._getvar("uvmet10", timeidx, units="knots"))
        except Exception:
            return

        surface_u = uv10[0]
        surface_v = uv10[1]
        try:
            u_1km = as_float_array(wrf.interplevel(uvmet[0], height_agl, 1000.0))
            v_1km = as_float_array(wrf.interplevel(uvmet[1], height_agl, 1000.0))
        except Exception:
            u_1km = None
            v_1km = None

        interpolated = []
        for level in HODO_OVERLAY_LEVELS_HPA:
            try:
                u_level = as_float_array(wrf.interplevel(uvmet[0], pressure, float(level)))
                v_level = as_float_array(wrf.interplevel(uvmet[1], pressure, float(level)))
                interpolated.append((u_level, v_level))
            except Exception:
                interpolated.append((None, None))

        lon_span = float(np.nanmax(self.lon) - np.nanmin(self.lon))
        lat_span = float(np.nanmax(self.lat) - np.nanmin(self.lat))
        if not np.isfinite(lon_span) or not np.isfinite(lat_span) or lon_span <= 0 or lat_span <= 0:
            return

        # Scale wind components into compact circular glyphs that resemble small standalone hodographs.
        size_scale = self.hodo_size_spin.value() / 10.0
        spacing_scale = self.hodo_spacing_spin.value()
        deg_per_kt_x = (lon_span / max(self.lon.shape[1], 1)) * 0.026 * size_scale
        deg_per_kt_y = (lat_span / max(self.lat.shape[0], 1)) * 0.026 * size_scale
        skip_y = max(6, self.lat.shape[0] // spacing_scale)
        skip_x = max(6, self.lat.shape[1] // spacing_scale)
        theta = np.linspace(0, 2 * np.pi, 120)
        ring_speeds = [20, 40]

        for j in range(skip_y // 2, self.lat.shape[0], skip_y):
            for i in range(skip_x // 2, self.lat.shape[1], skip_x):
                lon0 = float(self.lon[j, i])
                lat0 = float(self.lat[j, i])
                if not np.isfinite(lon0) or not np.isfinite(lat0):
                    continue

                points = []
                surface_u_val = float(surface_u[j, i])
                surface_v_val = float(surface_v[j, i])
                if np.isfinite(surface_u_val) and np.isfinite(surface_v_val):
                    points.append((surface_u_val, surface_v_val))

                if u_1km is not None and v_1km is not None:
                    u_1km_val = float(u_1km[j, i])
                    v_1km_val = float(v_1km[j, i])
                    if np.isfinite(u_1km_val) and np.isfinite(v_1km_val):
                        points.append((u_1km_val, v_1km_val))

                for u_grid, v_grid in interpolated:
                    if u_grid is None or v_grid is None:
                        continue
                    u_level = float(u_grid[j, i])
                    v_level = float(v_grid[j, i])
                    if np.isfinite(u_level) and np.isfinite(v_level):
                        points.append((u_level, v_level))

                if len(points) < 2:
                    continue

                x_points = [lon0 + u * deg_per_kt_x for u, _ in points]
                y_points = [lat0 + v * deg_per_kt_y for _, v in points]
                ring_radius_x = ring_speeds[-1] * deg_per_kt_x
                ring_radius_y = ring_speeds[-1] * deg_per_kt_y

                # Range rings
                for ring_speed in ring_speeds:
                    parent_axes.plot(
                        lon0 + np.cos(theta) * ring_speed * deg_per_kt_x,
                        lat0 + np.sin(theta) * ring_speed * deg_per_kt_y,
                        color="#2b2b2b",
                        linewidth=0.55 if ring_speed == ring_speeds[-1] else 0.45,
                        alpha=0.9 if ring_speed == ring_speeds[-1] else 0.75,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )

                # Center crosshair
                parent_axes.plot(
                    [lon0 - ring_radius_x * 0.18, lon0 + ring_radius_x * 0.18],
                    [lat0, lat0],
                    color="#333333",
                    linewidth=0.45,
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )
                parent_axes.plot(
                    [lon0, lon0],
                    [lat0 - ring_radius_y * 0.18, lat0 + ring_radius_y * 0.18],
                    color="#333333",
                    linewidth=0.45,
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )

                # Motion vector from center toward the first wind point
                parent_axes.plot(
                    [lon0, x_points[0]],
                    [lat0, y_points[0]],
                    color="#4c4c4c",
                    linewidth=0.85,
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )
                for idx in range(len(points) - 1):
                    parent_axes.plot(
                        x_points[idx:idx + 2],
                        y_points[idx:idx + 2],
                        color=HODO_OVERLAY_COLORS[min(idx, len(HODO_OVERLAY_COLORS) - 1)],
                        linewidth=1.0 + 0.06 * self.hodo_size_spin.value(),
                        solid_capstyle="round",
                        transform=ccrs.PlateCarree(),
                        zorder=7,
                    )
                parent_axes.plot(
                    lon0,
                    lat0,
                    marker="o",
                    markersize=1.2 + 0.05 * self.hodo_size_spin.value(),
                    color="#4a4a4a",
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                )

    def _create_axes(self, figure, field):
        if self.lat is not None and self.lon is not None and self.lat.shape == field.shape:
            return figure.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        return figure.subplots()

    def _set_map_background(self, axes, style, finite_values):
        if not hasattr(axes, "set_facecolor"):
            return

        try:
            if finite_values.size:
                background_value = float(np.nanmin(finite_values))
            elif style["levels"] is not None:
                background_value = float(style["levels"][0])
            else:
                background_value = 0.0

            norm = style.get("norm")
            cmap = style.get("cmap")
            if cmap is not None:
                color = cmap(norm(background_value) if norm is not None else 0.0)
                axes.set_facecolor(color)
                return
        except Exception:
            pass

        axes.set_facecolor("#1f2329")

    def _apply_state_borders(self, axes):
        if (
            self.state_borders_check.isChecked()
            and self.lat is not None
            and self.lon is not None
        ):
            axes.add_feature(STATES.with_scale("50m"), linewidth=0.5, edgecolor="black")

    def _draw_topography(self, figure, axes):
        if not self.topography_check.isChecked():
            return
        if not self.wrf_files or self.lat is None or self.lon is None:
            return

        try:
            terrain = as_float_array(self.wrf_files[0].getvar("terrain", timeidx=0, units="m"))
        except Exception:
            return

        if terrain.shape != self.lat.shape:
            return

        finite_terrain = terrain[np.isfinite(terrain)]
        if finite_terrain.size == 0:
            return

        terrain_mesh = axes.contourf(
            self.lon,
            self.lat,
            terrain,
            levels=MaxNLocator(nbins=12).tick_values(float(finite_terrain.min()), float(finite_terrain.max())),
            cmap="terrain",
            alpha=0.32,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        axes.contour(
            self.lon,
            self.lat,
            terrain,
            levels=8,
            colors="black",
            linewidths=0.22,
            alpha=0.35,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

        topo_colorbar = figure.colorbar(
            terrain_mesh,
            ax=axes,
            shrink=0.82,
            pad=0.02,
            fraction=0.05,
        )
        topo_colorbar.set_label("Terrain (m)")

    def _create_main_layout(self, figure):
        grid = figure.add_gridspec(
            1,
            3,
            width_ratios=[4.7, 0.3, 2.55],
            left=0.05,
            right=0.98,
            bottom=0.08,
            top=0.95,
            wspace=0.16,
        )
        map_axes = figure.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
        return map_axes, grid[0, 1], grid[0, 2]

    def _render_on_axes(self, figure, axes, field, timeidx, ndim, level, colorbar_spec=None):
        item = self.variable_combo.currentData() or {}
        units = self.selected_units_label() or display_units_name(item.get("units", ""))
        if self.selected_variable() == "maxdbz_uhel_fill":
            self._render_composite_reflectivity_uhel(figure, axes, timeidx, level, colorbar_spec, units)
            return
        use_old_dbz_rendering = self._use_old_dbz_rendering()
        smoothed = field if use_old_dbz_rendering else self._smoothed_field(field)
        use_smooth_render = self.smoothing_spin.value() > 0 and not use_old_dbz_rendering
        finite_values = smoothed[np.isfinite(smoothed)]
        style = self._field_style(finite_values)

        if self.lat is not None and self.lon is not None and self.lat.shape == smoothed.shape:
            self.map_axes = axes
            self._set_map_background(axes, style, finite_values)
            self._draw_topography(figure, axes)
            if use_old_dbz_rendering and finite_values.size:
                mesh = axes.contourf(
                    self.lon,
                    self.lat,
                    smoothed,
                    levels=style["levels"],
                    cmap=style["cmap"],
                    norm=style["norm"],
                    transform=ccrs.PlateCarree(),
                    antialiased=True,
                    extend=style["extend"],
                )
            elif use_smooth_render and finite_values.size:
                if style["levels"] is None:
                    mesh = axes.pcolormesh(
                        self.lon,
                        self.lat,
                        smoothed,
                        shading="gouraud",
                        cmap=style["cmap"],
                        norm=style["norm"],
                        transform=ccrs.PlateCarree(),
                    )
                else:
                    mesh = axes.contourf(
                        self.lon,
                        self.lat,
                        smoothed,
                        levels=style["levels"],
                        cmap=style["cmap"],
                        norm=style["norm"],
                        transform=ccrs.PlateCarree(),
                        antialiased=False,
                        extend=style["extend"],
                    )
            else:
                mesh = axes.pcolormesh(
                    self.lon,
                    self.lat,
                    smoothed,
                    shading="auto",
                    cmap=style["cmap"],
                    norm=style["norm"],
                    transform=ccrs.PlateCarree(),
                )
            axes.set_xlabel("Longitude")
            axes.set_ylabel("Latitude")
            self._apply_state_borders(axes)
            self._draw_wind_barbs(axes, timeidx)
            self._draw_selected_point(axes)
            self._draw_cross_section_selection(axes)
            self._render_hodograph_overlay(axes, timeidx)
        else:
            self.map_axes = None
            mesh = axes.imshow(
                smoothed,
                origin="lower",
                cmap=style["cmap"],
                norm=style["norm"],
                aspect="auto",
                interpolation="bilinear" if use_smooth_render else "nearest",
            )
            axes.set_xlabel("X")
            axes.set_ylabel("Y")

        title = f"{self.domain_label()}\n{self.selected_display_name()} | {self._field_time_label(timeidx)}"
        if level is not None:
            title += f" | level {level}"
        if self.smoothing_spin.value() > 0 and not use_old_dbz_rendering:
            title += f" | smooth {self.smoothing_spin.value()}"
        if self.topography_check.isChecked():
            title += " | topography"
        if self.wind_barbs_check.isChecked():
            title += (
                f" | barbs {self.wind_level_combo.currentText()}"
                f" size {self.wind_barb_size_spin.value()}"
                f" density {self.wind_barb_density_spin.value()}"
            )
        if self.hodograph_overlay_check.isChecked():
            title += f" | hodo size {self.hodo_size_spin.value()} spacing {self.hodo_spacing_spin.value()}"
        axes.set_title(title)
        self._add_field_colorbar(
            figure,
            mesh,
            style,
            units,
            axes=axes,
            colorbar_spec=colorbar_spec,
        )

    def _render_composite_reflectivity_uhel(self, figure, axes, timeidx, level, colorbar_spec, units):
        composite_reflectivity, uhel = self._fetch_composite_reflectivity_uhel_fields(timeidx)
        finite_values = composite_reflectivity[np.isfinite(composite_reflectivity)]
        style = self._field_style(finite_values)

        if self.lat is not None and self.lon is not None and self.lat.shape == composite_reflectivity.shape:
            self.map_axes = axes
            self._set_map_background(axes, style, finite_values)
            self._draw_topography(figure, axes)
            mesh = axes.contourf(
                self.lon,
                self.lat,
                composite_reflectivity,
                levels=style["levels"],
                cmap=style["cmap"],
                norm=style["norm"],
                transform=ccrs.PlateCarree(),
                antialiased=True,
                extend=style["extend"],
            )

            finite_uhel = uhel[np.isfinite(uhel)]
            if finite_uhel.size and float(np.nanmax(finite_uhel)) >= 75.0:
                upper_bound = max(76.0, float(np.nanmax(finite_uhel)) + 1.0)
                axes.contourf(
                    self.lon,
                    self.lat,
                    np.where(uhel >= 75.0, uhel, np.nan),
                    levels=[75.0, upper_bound],
                    colors=["#ff00ff"],
                    alpha=0.35,
                    transform=ccrs.PlateCarree(),
                    antialiased=True,
                    zorder=3,
                )

            axes.set_xlabel("Longitude")
            axes.set_ylabel("Latitude")
            self._apply_state_borders(axes)
            self._draw_wind_barbs(axes, timeidx)
            self._draw_selected_point(axes)
            self._draw_cross_section_selection(axes)
            self._render_hodograph_overlay(axes, timeidx)
        else:
            self.map_axes = None
            mesh = axes.imshow(
                composite_reflectivity,
                origin="lower",
                cmap=style["cmap"],
                norm=style["norm"],
                aspect="auto",
                interpolation="nearest",
            )
            finite_uhel = uhel[np.isfinite(uhel)]
            if finite_uhel.size and float(np.nanmax(finite_uhel)) >= 75.0:
                uh_mask = np.where(uhel >= 75.0, 1.0, np.nan)
                axes.imshow(
                    uh_mask,
                    origin="lower",
                    cmap=ListedColormap(["#ff00ff"]),
                    alpha=0.35,
                    aspect="auto",
                    interpolation="nearest",
                )
            axes.set_xlabel("X")
            axes.set_ylabel("Y")

        title = f"{self.domain_label()}\n{self.selected_display_name()} | {self.times[timeidx] if self.times else 'time 0'}"
        if level is not None:
            title += f" | level {level}"
        if self.topography_check.isChecked():
            title += " | topography"
        if self.wind_barbs_check.isChecked():
            title += (
                f" | barbs {self.wind_level_combo.currentText()}"
                f" size {self.wind_barb_size_spin.value()}"
                f" density {self.wind_barb_density_spin.value()}"
            )
        if self.hodograph_overlay_check.isChecked():
            title += f" | hodo size {self.hodo_size_spin.value()} spacing {self.hodo_spacing_spin.value()}"
        axes.set_title(title)

        self._add_field_colorbar(
            figure,
            mesh,
            style,
            units,
            axes=axes,
            colorbar_spec=colorbar_spec,
        )

    def _draw_field(self, figure, timeidx):
        field, ndim, level, level_count = self._fetch_field(timeidx)
        if self.hodograph_check.isChecked():
            axes, colorbar_spec, sounding_spec = self._create_main_layout(figure)
            self._render_on_axes(figure, axes, field, timeidx, ndim, level, colorbar_spec=colorbar_spec)
            self._render_sounding_panels(figure, sounding_spec, timeidx)
        else:
            axes = self._create_axes(figure, field)
            self._render_on_axes(figure, axes, field, timeidx, ndim, level)
        return ndim, level_count

    def _prime_style_range_for_times(self, time_indices):
        use_old_dbz_rendering = self._use_old_dbz_rendering()
        for timeidx in time_indices:
            field, _, _, _ = self._fetch_field(timeidx)
            styled_field = field if use_old_dbz_rendering else self._smoothed_field(field)
            finite_values = styled_field[np.isfinite(styled_field)]
            self._field_style(finite_values)

    def render_current_time(self):
        if not self.wrf_files:
            self._refresh_cross_section_window()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            preview, ndim, _, level_count = self._fetch_field(self._current_timeidx())
            if ndim == 3:
                self.level_spin.blockSignals(True)
                self.level_spin.setEnabled(True)
                self.level_spin.setMinimum(0)
                self.level_spin.setMaximum(level_count - 1)
                self.level_spin.setValue(min(self.level_spin.value(), level_count - 1))
                self.level_spin.blockSignals(False)
            else:
                self.level_spin.blockSignals(True)
                self.level_spin.setEnabled(False)
                self.level_spin.setMinimum(0)
                self.level_spin.setMaximum(0)
                self.level_spin.setValue(0)
                self.level_spin.blockSignals(False)

            self.map_axes = None
            self.figure.clear()
            if self.times:
                self.time_label.setText(
                    f"{self._current_timeidx() + 1}/{len(self.times)}: {self.times[self._current_timeidx()]}"
                )
            if self.hodograph_check.isChecked():
                axes, colorbar_spec, sounding_spec = self._create_main_layout(self.figure)
                self._render_on_axes(
                    self.figure,
                    axes,
                    preview,
                    self._current_timeidx(),
                    ndim,
                    self.level_spin.value() if ndim == 3 else None,
                    colorbar_spec=colorbar_spec,
                )
                self._render_sounding_panels(self.figure, sounding_spec, self._current_timeidx())
            else:
                axes = self._create_axes(self.figure, preview)
                self._render_on_axes(
                    self.figure,
                    axes,
                    preview,
                    self._current_timeidx(),
                    ndim,
                    self.level_spin.value() if ndim == 3 else None,
                )
            self.canvas.draw()
            self._refresh_cross_section_window()
            self.statusBar().showMessage("Rendered current frame.")
        except Exception as exc:
            self._refresh_cross_section_window()
            QMessageBox.critical(self, "Render failed", str(exc))
            self.statusBar().showMessage("Render failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def save_png(self):
        if not self.wrf_files:
            QMessageBox.information(self, "No file loaded", "Open a WRFOUT file before exporting.")
            return
        default_name = f"{self.selected_variable()}_{self._current_timeidx():03d}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", default_name, "PNG image (*.png)")
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=150)
            self.statusBar().showMessage(f"Saved PNG to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "PNG export failed", str(exc))

    def _build_sounding_profile(self, timeidx, point):
        if point is None:
            raise RuntimeError("No sounding point is selected.")

        j, i = point
        pressure = as_float_array(self._getvar("pressure", timeidx, units="hPa"))[:, j, i]
        height_agl = as_float_array(self._getvar("height_agl", timeidx, units="m"))[:, j, i]
        temperature = as_float_array(self._getvar("tc", timeidx))[:, j, i]
        dewpoint = as_float_array(self._getvar("td", timeidx))[:, j, i]
        uvmet = as_float_array(self._getvar("uvmet", timeidx, units="knots"))
        u_wind = uvmet[0, :, j, i]
        v_wind = uvmet[1, :, j, i]

        mask = (
            np.isfinite(pressure)
            & np.isfinite(height_agl)
            & np.isfinite(temperature)
            & np.isfinite(dewpoint)
            & np.isfinite(u_wind)
            & np.isfinite(v_wind)
        )
        if mask.sum() < 3:
            raise RuntimeError("No valid sounding profile was available for the selected point.")

        pressure = pressure[mask]
        height_agl = height_agl[mask]
        temperature = temperature[mask]
        dewpoint = dewpoint[mask]
        u_wind = u_wind[mask]
        v_wind = v_wind[mask]

        order = np.argsort(pressure)[::-1]
        return (
            pressure[order] * units.hectopascal,
            height_agl[order] * units.meter,
            temperature[order] * units.degC,
            dewpoint[order] * units.degC,
            u_wind[order] * units.knots,
            v_wind[order] * units.knots,
        )

    def _render_sounding_panels(self, figure, subplot_spec, timeidx):
        panel_grid = subplot_spec.subgridspec(2, 1, height_ratios=[4.2, 1.55], hspace=0.08)
        self._render_skew_t(figure, panel_grid[0, 0], timeidx)
        self._render_sounding_summary(figure, panel_grid[1, 0], timeidx)

    def _get_sounding_profile(self, timeidx):
        point = self._active_point()
        pressure, height_agl, temperature, dewpoint, u_wind, v_wind = self._build_sounding_profile(timeidx, point)
        j, i = point
        return pressure, height_agl, temperature, dewpoint, u_wind, v_wind, j, i

    def _style_sounding_axes(self, axes):
        axes.set_facecolor(SOUNDING_AXES_FACE)
        for spine in axes.spines.values():
            spine.set_color("#f5f5f5")
            spine.set_linewidth(1.0)
        axes.tick_params(colors=SOUNDING_TEXT, labelcolor=SOUNDING_TEXT, which="both")
        axes.xaxis.label.set_color(SOUNDING_TEXT)
        axes.yaxis.label.set_color(SOUNDING_TEXT)
        axes.title.set_color(SOUNDING_TEXT)

    def _wind_dir_speed(self, u_component, v_component):
        u_value = float(u_component.to("knots").m if hasattr(u_component, "to") else u_component)
        v_value = float(v_component.to("knots").m if hasattr(v_component, "to") else v_component)
        speed = float(np.hypot(u_value, v_value))
        direction = (270.0 - np.degrees(np.arctan2(v_value, u_value))) % 360.0
        return direction, speed

    def _interp_height_for_pressure(self, pressure, height_agl, target_pressure):
        pressure_values = pressure.to("hPa").m
        height_values = height_agl.to("meter").m
        ascending_pressure = pressure_values[::-1]
        ascending_height = height_values[::-1]
        target_value = float(target_pressure.to("hPa").m if hasattr(target_pressure, "to") else target_pressure)
        if target_value < float(np.nanmin(ascending_pressure)) or target_value > float(np.nanmax(ascending_pressure)):
            return np.nan
        return float(np.interp(target_value, ascending_pressure, ascending_height))

    def _find_effective_inflow_layer(self, pressure, height_agl, temperature, dewpoint):
        cape_threshold = 100.0
        cin_threshold = -250.0
        valid_indices = []

        for idx in range(len(pressure)):
            try:
                parcel_profile = mpcalc.parcel_profile(pressure[idx:], temperature[idx], dewpoint[idx]).to("degC")
                cape, cin = mpcalc.cape_cin(
                    pressure[idx:],
                    temperature[idx:],
                    dewpoint[idx:],
                    parcel_profile,
                )
                cape_value = float(cape.to("joule / kilogram").m)
                cin_value = float(cin.to("joule / kilogram").m)
                if np.isfinite(cape_value) and np.isfinite(cin_value):
                    if cape_value >= cape_threshold and cin_value >= cin_threshold:
                        valid_indices.append(idx)
            except Exception:
                continue

        if not valid_indices:
            return None

        contiguous = [valid_indices[0]]
        for idx in valid_indices[1:]:
            if idx == contiguous[-1] + 1:
                contiguous.append(idx)
            else:
                break

        if not contiguous:
            return None

        base_idx = contiguous[0]
        top_idx = contiguous[-1]
        return {
            "base_idx": base_idx,
            "top_idx": top_idx,
            "base_height_m": float(height_agl[base_idx].to("meter").m),
            "top_height_m": float(height_agl[top_idx].to("meter").m),
            "base_pressure_hpa": float(pressure[base_idx].to("hPa").m),
            "top_pressure_hpa": float(pressure[top_idx].to("hPa").m),
        }

    def _compute_sounding_parameters(self, pressure, height_agl, temperature, dewpoint, u_wind, v_wind):
        params = {}

        try:
            sbcape, sbcin = mpcalc.surface_based_cape_cin(pressure, temperature, dewpoint)
            params["sbcape"] = float(sbcape.to("joule / kilogram").m)
            params["sbcin"] = float(sbcin.to("joule / kilogram").m)
        except Exception:
            params["sbcape"] = np.nan
            params["sbcin"] = np.nan

        try:
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure, temperature, dewpoint)
            params["mlcape"] = float(mlcape.to("joule / kilogram").m)
            params["mlcin"] = float(mlcin.to("joule / kilogram").m)
        except Exception:
            params["mlcape"] = np.nan
            params["mlcin"] = np.nan

        try:
            mucape, mucin = mpcalc.most_unstable_cape_cin(pressure, temperature, dewpoint)
            params["mucape"] = float(mucape.to("joule / kilogram").m)
            params["mucin"] = float(mucin.to("joule / kilogram").m)
        except Exception:
            params["mucape"] = np.nan
            params["mucin"] = np.nan

        try:
            lcl_pressure, _ = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
            params["lcl_pressure"] = float(lcl_pressure.to("hPa").m)
            params["lcl_height"] = self._interp_height_for_pressure(pressure, height_agl, lcl_pressure)
        except Exception:
            params["lcl_pressure"] = np.nan
            params["lcl_height"] = np.nan

        try:
            lfc_pressure, _ = mpcalc.lfc(pressure, temperature, dewpoint)
            params["lfc_pressure"] = float(lfc_pressure.to("hPa").m)
            params["lfc_height"] = self._interp_height_for_pressure(pressure, height_agl, lfc_pressure)
        except Exception:
            params["lfc_pressure"] = np.nan
            params["lfc_height"] = np.nan

        try:
            el_pressure, _ = mpcalc.el(pressure, temperature, dewpoint)
            params["el_pressure"] = float(el_pressure.to("hPa").m)
            params["el_height"] = self._interp_height_for_pressure(pressure, height_agl, el_pressure)
        except Exception:
            params["el_pressure"] = np.nan
            params["el_height"] = np.nan

        try:
            pwat = mpcalc.precipitable_water(pressure, dewpoint)
            params["pwat"] = float(pwat.to("inch").m)
        except Exception:
            params["pwat"] = np.nan

        try:
            right_mover, left_mover, mean_wind = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, height_agl)
            params["rm_motion"] = self._wind_dir_speed(right_mover[0], right_mover[1])
            params["lm_motion"] = self._wind_dir_speed(left_mover[0], left_mover[1])
            params["mean_wind"] = self._wind_dir_speed(mean_wind[0], mean_wind[1])
            storm_u = right_mover[0]
            storm_v = right_mover[1]
        except Exception:
            params["rm_motion"] = (np.nan, np.nan)
            params["lm_motion"] = (np.nan, np.nan)
            params["mean_wind"] = (np.nan, np.nan)
            storm_u = 0 * units.knots
            storm_v = 0 * units.knots

        try:
            srh_0_1km, _, _ = mpcalc.storm_relative_helicity(
                height_agl,
                u_wind,
                v_wind,
                depth=1000 * units.meter,
                storm_u=storm_u,
                storm_v=storm_v,
            )
            params["srh_0_1km"] = float(srh_0_1km.to("meter ** 2 / second ** 2").m)
        except Exception:
            params["srh_0_1km"] = np.nan

        try:
            srh_0_3km, _, _ = mpcalc.storm_relative_helicity(
                height_agl,
                u_wind,
                v_wind,
                depth=3000 * units.meter,
                storm_u=storm_u,
                storm_v=storm_v,
            )
            params["srh_0_3km"] = float(srh_0_3km.to("meter ** 2 / second ** 2").m)
        except Exception:
            params["srh_0_3km"] = np.nan

        try:
            effective_layer = self._find_effective_inflow_layer(pressure, height_agl, temperature, dewpoint)
            params["effective_layer"] = effective_layer
            if effective_layer is not None:
                effective_bottom = effective_layer["base_height_m"] * units.meter
                esrh_0_1km, _, _ = mpcalc.storm_relative_helicity(
                    height_agl,
                    u_wind,
                    v_wind,
                    depth=1000 * units.meter,
                    bottom=effective_bottom,
                    storm_u=storm_u,
                    storm_v=storm_v,
                )
                params["effective_srh_0_1km"] = float(esrh_0_1km.to("meter ** 2 / second ** 2").m)

                esrh_0_3km, _, _ = mpcalc.storm_relative_helicity(
                    height_agl,
                    u_wind,
                    v_wind,
                    depth=3000 * units.meter,
                    bottom=effective_bottom,
                    storm_u=storm_u,
                    storm_v=storm_v,
                )
                params["effective_srh_0_3km"] = float(esrh_0_3km.to("meter ** 2 / second ** 2").m)
            else:
                params["effective_srh_0_1km"] = np.nan
                params["effective_srh_0_3km"] = np.nan
        except Exception:
            params["effective_layer"] = None
            params["effective_srh_0_1km"] = np.nan
            params["effective_srh_0_3km"] = np.nan

        try:
            shear_u, shear_v = mpcalc.bulk_shear(
                pressure,
                u_wind,
                v_wind,
                height=height_agl,
                depth=6000 * units.meter,
            )
            params["shear_0_6km"] = float(np.hypot(shear_u.to("knots").m, shear_v.to("knots").m))
        except Exception:
            params["shear_0_6km"] = np.nan

        return params

    def _format_param_value(self, value, fmt="{:.0f}", suffix=""):
        if value is None:
            return "--"
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(numeric):
            return "--"
        return fmt.format(numeric) + suffix

    def _format_motion_value(self, motion):
        if not motion or len(motion) != 2:
            return "--/-- kt"
        direction, speed = motion
        if not np.isfinite(direction) or not np.isfinite(speed):
            return "--/-- kt"
        return f"{direction:03.0f}/{speed:.0f} kt"

    def _render_sounding_summary(self, figure, subplot_spec, timeidx):
        try:
            pressure, height_agl, temperature, dewpoint, u_wind, v_wind, _, _ = self._get_sounding_profile(timeidx)
            params = self._compute_sounding_parameters(pressure, height_agl, temperature, dewpoint, u_wind, v_wind)
        except Exception as exc:
            axes = figure.add_subplot(subplot_spec)
            axes.text(0.5, 0.5, str(exc), ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            return

        axes = figure.add_subplot(subplot_spec)
        self._style_sounding_axes(axes)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.axhline(1.0, color="#00b7ff", linewidth=1.2)

        for xpos in [0.32, 0.67]:
            axes.axvline(xpos, color="#3c3f46", linewidth=0.8)

        header_color = "#f8fafc"
        label_color = "#93c5fd"
        value_color = "#facc15"
        font_family = "DejaVu Sans Mono"

        columns = [
            (
                0.03,
                0.19,
                "Parcel",
                [
                    ("SBCAPE", self._format_param_value(params.get("sbcape"))),
                    ("SBCIN", self._format_param_value(params.get("sbcin"))),
                    ("MLCAPE", self._format_param_value(params.get("mlcape"))),
                    ("MLCIN", self._format_param_value(params.get("mlcin"))),
                    ("MUCAPE", self._format_param_value(params.get("mucape"))),
                    ("MUCIN", self._format_param_value(params.get("mucin"))),
                    ("LCL", self._format_param_value(params.get("lcl_height"), "{:.0f}", " m")),
                    ("LFC", self._format_param_value(params.get("lfc_height"), "{:.0f}", " m")),
                    ("EL", self._format_param_value(params.get("el_height"), "{:.0f}", " m")),
                ],
            ),
            (
                0.35,
                0.56,
                "Kinematics",
                [
                    ("0-1km SRH", self._format_param_value(params.get("srh_0_1km"))),
                    ("0-3km SRH", self._format_param_value(params.get("srh_0_3km"))),
                    ("Eff 0-1 SRH", self._format_param_value(params.get("effective_srh_0_1km"))),
                    ("Eff 0-3 SRH", self._format_param_value(params.get("effective_srh_0_3km"))),
                    ("0-6km Shear", self._format_param_value(params.get("shear_0_6km"), "{:.0f}", " kt")),
                    ("RM", self._format_motion_value(params.get("rm_motion"))),
                    ("LM", self._format_motion_value(params.get("lm_motion"))),
                    ("MeanWind", self._format_motion_value(params.get("mean_wind"))),
                ],
            ),
            (
                0.695,
                0.90,
                "Moisture / Press",
                [
                    ("PWAT", self._format_param_value(params.get("pwat"), "{:.2f}", " in")),
                    ("SFC P", self._format_param_value(pressure[0].to("hPa").m, "{:.0f}", " mb")),
                    ("LCL P", self._format_param_value(params.get("lcl_pressure"), "{:.0f}", " mb")),
                    ("LFC P", self._format_param_value(params.get("lfc_pressure"), "{:.0f}", " mb")),
                    ("EL P", self._format_param_value(params.get("el_pressure"), "{:.0f}", " mb")),
                    (
                        "Eff Inflow",
                        "--"
                        if not params.get("effective_layer")
                        else (
                            f"{params['effective_layer']['base_height_m']:.0f}-"
                            f"{params['effective_layer']['top_height_m']:.0f} m"
                        ),
                    ),
                ],
            ),
        ]

        for x0, value_x, heading, rows in columns:
            axes.text(
                x0,
                0.9,
                heading,
                color=header_color,
                fontsize=8.5,
                fontweight="bold",
                fontfamily=font_family,
                transform=axes.transAxes,
            )
            y = 0.765
            for label, value in rows:
                axes.text(
                    x0,
                    y,
                    label,
                    color=label_color,
                    fontsize=7.7,
                    fontfamily=font_family,
                    transform=axes.transAxes,
                )
                axes.text(
                    value_x,
                    y,
                    value,
                    color=value_color,
                    fontsize=7.7,
                    ha="left",
                    fontfamily=font_family,
                    transform=axes.transAxes,
                )
                y -= 0.105

    def _render_skew_t(self, figure, subplot_spec, timeidx):
        try:
            pressure, height_agl, temperature, dewpoint, u_wind, v_wind, j, i = self._get_sounding_profile(timeidx)
        except Exception as exc:
            axes = figure.add_subplot(subplot_spec)
            axes.text(0.5, 0.5, str(exc), ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            return

        figure.patch.set_facecolor(SOUNDING_PANEL_FACE)
        skew = SkewT(figure, rotation=45, subplot=subplot_spec)
        self._style_sounding_axes(skew.ax)
        skew.ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#7a7a7a", alpha=0.25)
        skew.plot(pressure, temperature, color="#ff2020", linewidth=2.4, zorder=6)
        skew.plot(pressure, dewpoint, color="#00ff38", linewidth=2.4, zorder=6)
        try:
            parcel_profile = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to("degC")
            skew.plot(pressure, parcel_profile, color="#00e5ff", linewidth=1.3, linestyle="-", zorder=5)
            skew.shade_cape(pressure, temperature, parcel_profile, alpha=0.14, color="#ff00ff")
            skew.shade_cin(pressure, temperature, parcel_profile, dewpoint, alpha=0.1, color="#00bfff")
        except Exception:
            parcel_profile = None
        barb_stride = max(2, len(pressure) // 11)
        skew.plot_barbs(
            pressure[::barb_stride],
            u_wind[::barb_stride],
            v_wind[::barb_stride],
            xloc=1.12,
            linewidth=0.5,
            length=5.2,
            color=SOUNDING_TEXT,
        )
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 50)
        skew.ax.set_xticks(np.arange(-40, 51, 10))
        skew.ax.set_yticks([1000, 925, 850, 700, 500, 300, 200, 100])
        skew.plot_dry_adiabats(alpha=0.2, linewidth=0.55, colors="#666666")
        skew.plot_moist_adiabats(alpha=0.22, linewidth=0.55, colors="#2a55ff")
        skew.plot_mixing_lines(alpha=0.16, linewidth=0.55, colors="#008f20")
        for temperature_line in range(-100, 61, 10):
            skew.ax.axvline(
                temperature_line,
                color="#cfcfcf",
                linewidth=0.55,
                linestyle="--",
                alpha=0.3,
                zorder=0,
            )
        skew.ax.set_xlabel("Temperature (C)")
        skew.ax.set_ylabel("Pressure (mb)")

        axes = skew.ax
        axes.set_title(
            f"Skew-T / Hodograph\n{self.lat[j, i]:.2f}, {self.lon[j, i]:.2f}",
            fontsize=10,
        )
        transform = blended_transform_factory(axes.transAxes, axes.transData)
        for height_m in [1000, 3000, 6000, 9000, 12000, 15000]:
            idx = int(np.argmin(np.abs(height_agl.to("meter").m - height_m)))
            pressure_val = pressure[idx].m
            if 100 <= pressure_val <= 1000:
                axes.text(
                    0.015,
                    pressure_val,
                    f"{int(height_m / 1000)} km",
                    fontsize=7,
                    color="#ff5252",
                    va="center",
                    ha="left",
                    transform=transform,
                )
                axes.hlines(
                    pressure_val,
                    xmin=0.0,
                    xmax=0.06,
                    colors="#ff5252",
                    linewidth=0.8,
                    transform=transform,
                )

        if self.side_hodograph_check.isChecked():
            inset = axes.inset_axes([0.58, 0.5, 0.38, 0.42])
            self._render_hodograph_axes(
                inset,
                pressure,
                height_agl,
                u_wind,
                v_wind,
                j,
                i,
                compact=True,
            )

    def _render_hodograph_axes(self, axes, pressure, height_agl, u_wind, v_wind, j, i, compact=False):
        self._style_sounding_axes(axes)
        axes.set_aspect("equal", adjustable="box")
        wind_u = u_wind.to("knots").m
        wind_v = v_wind.to("knots").m
        height_values = height_agl.to("meter").m
        finite_mask = np.isfinite(wind_u) & np.isfinite(wind_v) & np.isfinite(height_values)
        wind_u = wind_u[finite_mask]
        wind_v = wind_v[finite_mask]
        pressure_values = pressure.m[finite_mask]
        height_values = height_values[finite_mask]
        if wind_u.size < 2:
            axes.text(0.5, 0.5, "Not enough valid wind data", ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            return

        max_component = float(np.nanmax(np.abs(np.concatenate([wind_u, wind_v]))))
        component_range = max(40, int(np.ceil(max_component / 10.0) * 10) + 10)
        hodo = Hodograph(axes, component_range=component_range)
        hodo.add_grid(increment=10 if compact else 20)
        layer_edges = [0, 1000, 3000, 6000, 9000, 12000, 16000]
        for idx in range(len(layer_edges) - 1):
            mask = (height_values >= layer_edges[idx]) & (height_values <= layer_edges[idx + 1])
            if mask.sum() < 2:
                continue
            hodo.plot(
                wind_u[mask],
                wind_v[mask],
                color=HODO_SEGMENT_COLORS[min(idx, len(HODO_SEGMENT_COLORS) - 1)],
                linewidth=2.5,
            )
        marker_stride = max(1, wind_u.size // 8)
        axes.scatter(wind_u[::marker_stride], wind_v[::marker_stride], s=10, color="#e5e7eb", zorder=4)
        for target_height in [1000, 3000, 6000, 9000]:
            idx = int(np.argmin(np.abs(height_values - target_height)))
            axes.text(
                wind_u[idx] + 1.5,
                wind_v[idx] + 1.5,
                f"{int(target_height / 1000)}",
                fontsize=7,
                color="#d1d5db",
            )
        axes.scatter(
            wind_u[0],
            wind_v[0],
            s=22,
            color="#dc2626",
            zorder=5,
        )
        axes.axhline(0, color="#f5f5f5", linewidth=0.8, linestyle="-", alpha=0.9)
        axes.axvline(0, color="#f5f5f5", linewidth=0.8, linestyle="-", alpha=0.9)
        tick_step = 10 if compact else 20
        ticks = np.arange(-component_range, component_range + tick_step, tick_step)
        axes.set_xticks(ticks)
        axes.set_yticks(ticks)
        axes.grid(True, linestyle="--", linewidth=0.5, color="#8a8a8a", alpha=0.28)
        if compact:
            axes.set_title("Hodograph", fontsize=9, pad=3)
            axes.set_xlabel("")
            axes.set_ylabel("")
            axes.tick_params(labelsize=7, pad=1)
        else:
            axes.set_title(
                f"Hodograph\n{self.lat[j, i]:.2f}, {self.lon[j, i]:.2f}",
                fontsize=10,
            )
            axes.set_xlabel("U wind (kt)")
            axes.set_ylabel("V wind (kt)")

    def _render_hodograph(self, figure, subplot_spec, timeidx):
        try:
            pressure, height_agl, _, _, u_wind, v_wind, j, i = self._get_sounding_profile(timeidx)
        except Exception as exc:
            axes = figure.add_subplot(subplot_spec)
            axes.text(0.5, 0.5, str(exc), ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            return

        axes = figure.add_subplot(subplot_spec)
        self._render_hodograph_axes(axes, pressure, height_agl, u_wind, v_wind, j, i, compact=False)

    def export_gif(self):
        if not self.wrf_files or not self.times:
            QMessageBox.information(self, "No file loaded", "Open a WRFOUT file before exporting.")
            return

        time_range = self._prompt_gif_time_range()
        if time_range is None:
            return

        start_timeidx, end_timeidx = time_range
        default_name = f"{self.selected_variable()}.gif"
        path, _ = QFileDialog.getSaveFileName(self, "Export GIF", default_name, "GIF image (*.gif)")
        if not path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        frames = []
        export_indices = list(range(start_timeidx, end_timeidx + 1))
        try:
            self.statusBar().showMessage("Scanning GIF color range...")
            QApplication.processEvents()
            self._prime_style_range_for_times(export_indices)
            for frame_number, timeidx in enumerate(export_indices, start=1):
                self.statusBar().showMessage(
                    f"Rendering GIF frame {frame_number}/{len(export_indices)} "
                    f"(time {timeidx + 1}/{len(self.times)})..."
                )
                QApplication.processEvents()
                figure = Figure(figsize=(12, 7))
                self._draw_field(figure, timeidx)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    frames.append(image.convert("P", palette=Image.ADAPTIVE).copy())

            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=500,
                loop=0,
            )
            self.statusBar().showMessage(f"Saved GIF to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "GIF export failed", str(exc))
            self.statusBar().showMessage("GIF export failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def _prompt_gif_time_range(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("GIF Time Range")

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        start_spin = QSpinBox(dialog)
        start_spin.setMinimum(1)
        start_spin.setMaximum(len(self.times))
        start_spin.setValue(self._current_timeidx() + 1)
        form.addRow("Start time", start_spin)

        end_spin = QSpinBox(dialog)
        end_spin.setMinimum(1)
        end_spin.setMaximum(len(self.times))
        end_spin.setValue(len(self.times))
        form.addRow("End time", end_spin)

        time_hint = QLabel(
            f"Available timesteps: 1-{len(self.times)}\n"
            f"Start: {self.times[0]}\n"
            f"End: {self.times[-1]}"
        )
        time_hint.setWordWrap(True)
        form.addRow("Range", time_hint)

        def sync_start_limit(value):
            start_spin.setMaximum(value)
            if start_spin.value() > value:
                start_spin.setValue(value)

        def sync_end_limit(value):
            end_spin.setMinimum(value)
            if end_spin.value() < value:
                end_spin.setValue(value)

        start_spin.valueChanged.connect(sync_end_limit)
        end_spin.valueChanged.connect(sync_start_limit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None

        return start_spin.value() - 1, end_spin.value() - 1


def main():
    app = QApplication(sys.argv)
    window = WrfViewer()
    window.show()

    if len(sys.argv) > 1:
        launch_path = Path(sys.argv[1])
        if launch_path.is_dir():
            window.load_folder(str(launch_path))
        else:
            window.load_file(str(launch_path))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
