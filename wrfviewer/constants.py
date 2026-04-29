DISPLAY_NAME_OVERRIDES = {
    "qv": "Water Vapor Mixing Ratio",
    "qc": "Cloud Water Mixing Ratio",
    "qr": "Rain Water Mixing Ratio",
    "qi": "Ice Mixing Ratio",
    "qs": "Snow Mixing Ratio",
    "qg": "Graupel Mixing Ratio",
    "QCLOUD": "Cloud Water Mixing Ratio",
    "QRAIN": "Rain Water Mixing Ratio",
    "QICE": "Ice Mixing Ratio",
    "QSNOW": "Snow Mixing Ratio",
    "SWDOWN": "Downward Shortwave Flux at Surface",
    "SWNORM": "Normal Incident Shortwave Flux",
    "ALBEDO": "Surface Albedo",
    "planetary_albedo": "Planetary Albedo",
    "SWDNB": "Downward Shortwave Flux at Bottom",
    "SWDNBC": "Clear-Sky Downward Shortwave Flux at Bottom",
    "SWUPB": "Upward Shortwave Flux at Bottom",
    "SWUPBC": "Clear-Sky Upward Shortwave Flux at Bottom",
    "SWDNT": "Downward Shortwave Flux at Top",
    "SWDNTC": "Clear-Sky Downward Shortwave Flux at Top",
    "SWUPT": "Upward Shortwave Flux at Top",
    "SWUPTC": "Clear-Sky Upward Shortwave Flux at Top",
    "RTHRATSW": "Shortwave Heating Rate",
    "surface_precip": "Surface Precipitation (RAINC + RAINNC)",
    "pblh": "PBL Depth",
    "tke": "Turbulent Kinetic Energy",
    "hfx": "Sensible Heat Flux",
    "lh": "Latent Heat Flux",
    "qfx": "Moisture Flux",
    "dbz": "10cm Reflectivity",
    "maxdbz": "Composite 10cm Reflectivity",
    "maxdbz_uhel_fill": "Composite Reflectivity & 1-hr UH > 75 m^2 s^-2",
}

BARB_LEVEL_OPTIONS = [
    ("Surface (10 m)", "surface"),
    ("1000 mb", 1000),
    ("925 mb", 925),
    ("850 mb", 850),
    ("700 mb", 700),
    ("500 mb", 500),
    ("300 mb", 300),
]

HODO_OVERLAY_LEVELS_HPA = [925, 850, 700, 500, 300]
HODO_OVERLAY_COLORS = ["#fff7c2", "#9ef0de", "#41c7d9", "#1e73d8", "#1e73d8", "#ff7a12"]
SOUNDING_PANEL_FACE = "#202225"
SOUNDING_AXES_FACE = "#26282b"
SOUNDING_GRID = "#6b7280"
SOUNDING_TEXT = "#f3f4f6"
HODO_SEGMENT_COLORS = ["#ff00ff", "#ff4d4d", "#ffb000", "#ffff00", "#8cff00", "#00e5ff"]

PIVOTAL_DBZ_LEVELS = [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
PIVOTAL_DBZ_COLORS = [
    "#646464",
    "#04e9e7",
    "#019ff4",
    "#0300f4",
    "#02fd02",
    "#01c501",
    "#008e00",
    "#fdf802",
    "#e5bc00",
    "#fd9500",
    "#fd0000",
    "#d40000",
    "#bc0000",
    "#f800fd",
    "#9854c6",
    "#fdfdfd",
]

PIVOTAL_DEWPOINT_LEVELS_C = [-30, -20, -10, 0, 5, 10, 13, 16, 18, 20, 22, 24, 26, 28]
PIVOTAL_DEWPOINT_COLORS = [
    "#6e2ca3",
    "#3c4ec2",
    "#1f8fe5",
    "#30cfd0",
    "#3fe37a",
    "#9be539",
    "#d8e93d",
    "#ffd34d",
    "#ffaf38",
    "#ff8333",
    "#ff5a36",
    "#f03b2d",
    "#cf1b1b",
]

PIVOTAL_TEMP_LEVELS_F = [-30, -20, -10, 0, 10, 20, 32, 40, 50, 60, 70, 80, 90, 100, 110]
PIVOTAL_TEMP_COLORS = [
    "#f4d6ff",
    "#d5b3ff",
    "#a78bfa",
    "#6ea8ff",
    "#57d3ff",
    "#8cf7ff",
    "#e7f7ff",
    "#99e265",
    "#5bcc46",
    "#e6d84a",
    "#ffb347",
    "#ff7f32",
    "#ff4b2b",
    "#d7191c",
]
