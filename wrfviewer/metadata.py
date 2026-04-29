import re


VARIABLE_CATEGORY_ORDER = [
    "Temperature",
    "Moisture",
    "Hydrometeors & Surface Precipitation",
    "Boundary Layer & Fluxes",
    "Wind",
    "Pressure & Height",
    "Instability & Parcels",
    "Severe Storm Diagnostics",
    "Clouds & Precipitation",
    "Fire Weather",
    "Geography & Grid",
    "Other",
]


def categorize_variable(item):
    name = item["name"].lower()
    description = item["description"].lower()
    text = f"{name} {description}"
    words = set(re.findall(r"[a-z0-9_+-]+", text))

    if name in {"qv", "qc", "qr", "qi", "qs", "qg", "qcloud", "qrain", "qice", "qsnow", "surface_precip"}:
        return "Hydrometeors & Surface Precipitation"

    if (
        name in {"pblh", "tke", "hfx", "lh", "qfx"}
        or "boundary layer" in text
        or "pbl" in words
    ):
        return "Boundary Layer & Fluxes"

    if any(
        token in words or token in text
        for token in [
            "shortwave",
            "longwave",
            "radiation",
            "radiative",
            "heating rate",
            "flux",
        ]
    ) or name in {
        "swdown",
        "swnorm",
        "albedo",
        "planetary_albedo",
        "swdnb",
        "swdnbc",
        "swupb",
        "swupbc",
        "swdnt",
        "swdntc",
        "swupt",
        "swuptc",
        "rthratsw",
    }:
        return "Clouds & Precipitation"

    if name in {"lat", "lon", "terrain", "geopt"} or "grid" in words:
        return "Geography & Grid"

    if (
        "fire weather" in text
        or name in {"fosberg", "haines", "hdw"}
        or "hot-dry-windy" in text
    ):
        return "Fire Weather"

    if any(
        token in words or token in text
        for token in [
            "reflectivity",
            "cloud",
            "precipitable water",
            "cloud-top",
        ]
    ) or name in {"dbz", "maxdbz", "ctt", "cloudfrac", "pw", "uhel"}:
        return "Clouds & Precipitation"

    if any(
        token in words or token in text
        for token in [
            "tornado",
            "supercell",
            "helicity",
            "hail",
            "critical angle",
            "bulk richardson",
            "storm-relative",
            "bunkers",
            "shear",
        ]
    ) or name in {
        "srh1",
        "srh3",
        "srh",
        "shear_0_1km",
        "shear_0_6km",
        "bunkers_rm",
        "bunkers_lm",
        "mean_wind_0_6km",
        "stp",
        "stp_fixed",
        "stp_effective",
        "scp",
        "ehi",
        "critical_angle",
        "ship",
        "bri",
        "effective_srh",
        "bulk_shear",
        "mean_wind",
    }:
        return "Severe Storm Diagnostics"

    if any(
        token in words or token in text
        for token in [
            "cape",
            "cin",
            "parcel",
            "lcl",
            "lfc",
            "equilibrium level",
            "effective inflow",
            "lapse rate",
        ]
    ) or name in {
        "sbcape",
        "sbcin",
        "mlcape",
        "mlcin",
        "mucape",
        "mucin",
        "lcl",
        "lfc",
        "el",
        "cape2d",
        "cape3d",
        "cape",
        "cin",
        "effective_inflow",
        "effective_cape",
        "lapse_rate_700_500",
        "lapse_rate_0_3km",
        "lapse_rate",
    }:
        return "Instability & Parcels"

    if (
        "pressure" in words
        or "sea-level pressure" in text
        or "height" in words
        or name in {"pressure", "height", "height_agl", "omega", "avo", "pvo", "slp"}
    ):
        return "Pressure & Height"

    if (
        "wind" in words
        or name in {"ua", "va", "wa", "uvmet", "uvmet10", "wspd", "wdir", "wspd10", "wdir10"}
    ):
        return "Wind"

    if any(
        token in words or token in text
        for token in [
            "dewpoint",
            "humidity",
            "mixing ratio",
            "wet-bulb",
            "water vapor",
        ]
    ) or name in {"td", "dp2m", "rh", "rh2m", "mixing_ratio", "specific_humidity", "twb", "theta_w"}:
        return "Moisture"

    if (
        "temperature" in words
        or "virtual temperature" in text
        or name in {"t2", "tv2m", "temp", "tc", "theta", "theta_e", "tv", "freezing_level", "wet_bulb_0"}
    ):
        return "Temperature"

    return "Other"
