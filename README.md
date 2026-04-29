# WRFOUT Viewer

Desktop viewer for WRFOUT files built with `PySide6`, `matplotlib`, and `wrf-rust`.

## Features

- Open a single `wrfout_*` file or load all matching WRF files from a folder
- Browse all diagnostics exposed by `wrf-rust`
- Move through forecast times with a slider
- Inspect 3-D fields with a vertical level selector
- Toggle a terrain/topography underlay on geographic maps
- Toggle wind barbs at the surface or standard pressure levels
- Toggle an inline Skew-T panel and click the map to choose its sounding point
- Toggle map-embedded hodograph overlays on top of any map product
- Adjust map hodograph overlay size and spacing from the sidebar
- Set the vertical cross-section top height from the sidebar
- Browse PBL and surface flux variables in the main viewer (`pblh`, `tke`, `hfx`, `lh`, `qfx`)
- Export the current frame to PNG
- Export all timesteps of the current variable to GIF

## Run

Install the Python package if needed:

```powershell
pip install --user wrf-rust
```

Start the viewer:

```powershell
python .\wrf_viewer.py
```

Start the PBL-focused viewer:

```powershell
python .\wrf_viewer_pbl.py
```

You can also open a file or folder at launch:

```powershell
python .\wrf_viewer.py D:\path\to\wrfout_d01_2024-06-01_00_00_00
```

```powershell
python .\wrf_viewer.py D:\path\to\WRFout
```

## Notes

- The variable list comes from `wrf.list_variables()` and is split into category and variable dropdowns in the sidebar.
- `Open Folder` scans the selected directory for `wrfout*`, `.nc`, and `.cdf` files, skips unreadable entries, and stitches all discovered timesteps into one slider sequence.
- `dbz` is shown as `10cm Reflectivity`, and `maxdbz` as `Composite 10cm Reflectivity`.
- `dbz`/`maxdbz`, dewpoint (`td`, `dp2m`), and temperature (`tc`, `t2`, `temp`) now use vendored `autumnplot-gl` color tables from [`src/json`](https://github.com/tsupinie/autumnplot-gl/tree/main/src/json) by default.
- Color scales stay fixed per variable/unit while browsing, and automatically expand if the loaded WRF data falls outside the current table range.
- Unit overrides are passed through to `wrf.getvar(..., units=...)`.
- Wind barbs use `uvmet10` for the surface and `uvmet` interpolated to the selected pressure level for upper-air plots.
- Clicking the map updates the selected point marker used by the Skew-T panel.
- The hodograph overlay draws sparse, colored mini hodographs directly on the map to highlight wind profile curvature over the domain.
- `Hodo size` changes the overlay trace length and now defaults to `20`, while `Hodo spacing` changes how densely the mini hodographs are drawn.
- The overlay includes an explicit surface-to-1 km AGL segment in red before continuing through the upper-level trace.
- GIF export renders every available timestep for the currently selected variable and level.
