import json
from pathlib import Path

import numpy as np


AUTUMNPLOT_JSON_DIR = Path(__file__).resolve().parent.parent / "colortables" / "autumnplot_gl"
AUTUMNPLOT_JSON_FILES = {
    "dbz": "nws_storm_clear_refl_colormap.json",
    "maxdbz": "nws_storm_clear_refl_colormap.json",
    "maxdbz_uhel_fill": "nws_storm_clear_refl_colormap.json",
    "t2": "pwt2m_colormap.json",
    "temp": "pwt2m_colormap.json",
    "tc": "pwt2m_colormap.json",
    "dp2m": "pwtd2m_colormap.json",
    "td": "pwtd2m_colormap.json",
}
AUTUMNPLOT_CACHE = {}


def as_float_array(data):
    array = np.asanyarray(data)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)
    return np.asarray(array, dtype=float)


def f_to_c(values_f):
    return [(value - 32.0) * 5.0 / 9.0 for value in values_f]


def f_to_k(values_f):
    return [((value - 32.0) * 5.0 / 9.0) + 273.15 for value in values_f]


def display_units_name(units_name):
    return "mb" if units_name == "hPa" else units_name


def request_units_name(units_name):
    if units_name is None:
        return None
    return "hPa" if units_name.strip().lower() == "mb" else units_name


def load_autumnplot_colortable(name):
    if name in AUTUMNPLOT_CACHE:
        return AUTUMNPLOT_CACHE[name]

    path = AUTUMNPLOT_JSON_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    levels = [float(value) for value in data["levels"]]
    colors = [str(value) for value in data["colors"]]
    if len(levels) != len(colors) + 1:
        raise ValueError(f"{path} must contain exactly one more level than colors")

    table = {"levels": levels, "colors": colors}
    AUTUMNPLOT_CACHE[name] = table
    return table


def convert_autumnplot_levels(levels, units_name):
    normalized = (units_name or "").strip().lower()
    if "f" in normalized:
        return list(levels)
    if normalized in {"k", "kelvin"}:
        return [((value - 32.0) * 5.0 / 9.0) + 273.15 for value in levels]
    return [(value - 32.0) * 5.0 / 9.0 for value in levels]
