#!/usr/bin/env python3
"""
MapAltura — Visualización de numeraciones de calles en Mendoza, Argentina.

Lee los addr:housenumber directamente del PBF de Argentina,
los asigna a cada segmento de calle y genera un mapa HTML interactivo.

Uso:
    python main.py

Salida:
    output/mapa_alturas.html  → abrir en cualquier navegador web
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import geopandas as gpd
import folium
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import pbf_reader

# ── Configuración ──────────────────────────────────────────────────────────────

PBF_FILE    = "argentina-260823.osm.pbf"
COLORMAP    = "turbo"       # plasma | viridis | turbo | RdYlGn
OUTPUT_DIR  = "output"
TILE_STYLE  = "CartoDB dark_matter"
SEARCH_R    = 80             # metros: radio de búsqueda de alturas por segmento

REGIONS = [
    ("gran_mendoza", 32719, {
        "minlon": -68.91, "minlat": -33.09,
        "maxlon": -68.65, "maxlat": -32.77,
    }),
    ("buenos_aires", 32721, {
        "minlon": -58.55, "minlat": -34.75,
        "maxlon": -58.35, "maxlat": -34.55,
    }),
    ("cordoba", 32720, {
        "minlon": -64.27, "minlat": -31.47,
        "maxlon": -64.10, "maxlat": -31.35,
    }),
    ("rosario", 32720, {
        "minlon": -60.72, "minlat": -33.01,
        "maxlon": -60.61, "maxlat": -32.87,
    }),
]

# ── 1. Extracción de alturas desde el PBF ──────────────────────────────────────


def extract_addresses_from_pbf(pbf_path: str, bbox: dict) -> list:
    print("[1/4] Leyendo alturas del archivo PBF...")
    print(f"      lon [{bbox['minlon']}, {bbox['maxlon']}]  "
          f"lat [{bbox['minlat']}, {bbox['maxlat']}]")
    if not os.path.exists(pbf_path):
        sys.exit(f"No se encontró el archivo: {pbf_path}")
    records = pbf_reader.extract_addresses(pbf_path, bbox)
    print(f"      {len(records):,} puntos de numeración extraídos.")
    if not records:
        sys.exit(
            "No se encontraron numeraciones en el área. "
            "Verificá que el BBOX corresponda a una zona con datos."
        )
    return records


# ── 2. Red vial ─────────────────────────────────────────────────────────────────

def download_street_network(bbox: dict) -> gpd.GeoDataFrame:
    print("[2/4] Leyendo red vial del PBF (sin internet)...")
    edges = pbf_reader.extract_street_network(PBF_FILE, bbox)
    edges = edges.explode(index_parts=False).reset_index(drop=True)
    print(f"      {len(edges):,} segmentos de calle extraídos.")
    return edges


# ── 3. Asignación de alturas a segmentos ────────────────────────────────────────

def assign_heights(edges: gpd.GeoDataFrame, records: list, epsg_local: int) -> np.ndarray:
    """
    Para cada segmento calcula la mediana de los housenumbers OSM
    que estén a menos de SEARCH_R metros de su centroide.
    """
    print("[3/4] Asignando alturas a segmentos de calle...")

    edges_proj = edges.to_crs(epsg=epsg_local)

    addr_gdf = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(
            [r["lon"] for r in records],
            [r["lat"] for r in records]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=epsg_local)

    coords  = np.array([[p.x, p.y] for p in addr_gdf.geometry])
    numbers = np.array([r["number"] for r in records], dtype=float)
    tree    = cKDTree(coords)

    heights = np.full(len(edges_proj), np.nan)
    for i, geom in enumerate(edges_proj.geometry):
        if geom is None or geom.is_empty:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y
        idxs = tree.query_ball_point([cx, cy], r=SEARCH_R)
        if idxs:
            heights[i] = float(np.median(numbers[idxs]))

    covered = int(np.sum(~np.isnan(heights)))
    pct = covered / len(heights) * 100
    print(f"      {covered:,}/{len(heights):,} segmentos con datos ({pct:.1f}%).")

    if covered == 0:
        sys.exit(
            "Ningún segmento recibió alturas. "
            "Probá aumentar SEARCH_R o ajustar el BBOX."
        )
    return heights


# ── 4. Mapa Folium ──────────────────────────────────────────────────────────────

def build_map(edges: gpd.GeoDataFrame, heights: np.ndarray) -> folium.Map:
    """Construye el mapa interactivo coloreando cada segmento por su altura."""
    print("[4/4] Generando mapa interactivo...")

    valid = heights[~np.isnan(heights)]
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    try:
        cmap = matplotlib.colormaps[COLORMAP]
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(COLORMAP)

    def to_hex(v: float) -> str:
        if np.isnan(v):
            return "#2a2a2a"
        return mcolors.to_hex(cmap(norm(v)))

    center_lat = edges.geometry.centroid.y.mean()
    center_lon = edges.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=TILE_STYLE)

    for i, (_, row) in enumerate(edges.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        color = to_hex(heights[i])
        coords = [(y, x) for x, y in geom.coords]
        if len(coords) < 2:
            continue

        name = row.get("name")
        if isinstance(name, list):
            name = " / ".join(str(n) for n in name)
        name = name or "Sin nombre"

        h = heights[i]
        h_str = f"~{int(h)}" if not np.isnan(h) else "sin datos"

        folium.PolyLine(
            coords,
            color=color,
            weight=3.5,
            opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b>{name}</b><br>Altura: <b>{h_str}</b>",
                sticky=False
            )
        ).add_to(m)

    m.get_root().html.add_child(folium.Element(_legend_html(vmin, vmax, cmap)))
    return m


def _legend_html(vmin: float, vmax: float, cmap) -> str:
    steps = 12
    stops = ", ".join(
        f"{mcolors.to_hex(cmap(i / (steps - 1)))} {round(i / (steps - 1) * 100)}%"
        for i in range(steps)
    )
    ticks = [int(vmin + (vmax - vmin) * t) for t in (0, 0.25, 0.5, 0.75, 1.0)]
    ticks_html = "".join(
        f'<span style="position:absolute;left:{round((v - vmin) / (vmax - vmin) * 100)}%;'
        f'transform:translateX(-50%)">{v}</span>'
        for v in ticks
    )
    return f"""
<div style="
  position:fixed;bottom:40px;right:15px;z-index:9999;
  background:rgba(15,15,15,0.92);padding:14px 16px;border-radius:10px;
  color:#eee;font-family:Arial,sans-serif;font-size:13px;width:230px;
  box-shadow:0 2px 14px rgba(0,0,0,0.7);">
  <div style="font-size:15px;font-weight:bold;margin-bottom:3px;">Altura de calle</div>
  <div style="font-size:11px;color:#999;margin-bottom:9px;">Numeración según datos OSM</div>
  <div style="height:14px;border-radius:4px;
              background:linear-gradient(to right,{stops});"></div>
  <div style="position:relative;height:18px;font-size:10px;color:#bbb;margin-top:2px;">
    {ticks_html}
  </div>
  <div style="margin-top:6px;font-size:10px;color:#555;">Datos: © OpenStreetMap contributors</div>
</div>
"""


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  MapAltura — Numeraciones de calles de Mendoza (PBF)")
    print("=" * 58)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for region_name, epsg_local, bbox in REGIONS:
        output_file = os.path.join(OUTPUT_DIR, f"mapa_altura_{region_name}.html")
        print(f"\n--- {region_name.replace('_', ' ').title()} ---")
        records = extract_addresses_from_pbf(PBF_FILE, bbox)
        edges   = download_street_network(bbox)
        heights = assign_heights(edges, records, epsg_local)
        m       = build_map(edges, heights)
        m.save(output_file)

        valid = heights[~np.isnan(heights)]
        print(f"  Mapa generado : {output_file}")
        print(f"  Alturas reales: {int(valid.min())} – {int(valid.max())}")
        print(f"  Puntos OSM    : {len(records):,}")


if __name__ == "__main__":
    main()
