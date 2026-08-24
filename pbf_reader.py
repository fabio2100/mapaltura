"""
pbf_reader.py — Lector puro-Python de archivos OSM PBF.

Sin dependencias externas (solo struct, zlib, io de stdlib).

Formato PBF de OSM:
  File  = (4-byte big-endian BlobHeader length | BlobHeader | Blob)*
  Blob  = raw bytes | zlib_data (deflate)
  PrimitiveBlock → StringTable + PrimitiveGroup[]
  PrimitiveGroup → DenseNodes | Node[] | Way[] | Relation[]
  DenseNodes → id[], lat[], lon[] (delta/zigzag) + keys_vals (pares key/val, sep=0)
"""

import os
import io
import struct
import zlib


# ── Protobuf mínimo ───────────────────────────────────────────────────────────

def _varint(s: io.BytesIO):
    """Lee un varint de un BytesIO. Devuelve None en EOF."""
    result = shift = 0
    while True:
        b = s.read(1)
        if not b:
            return None
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result
        shift += 7


def _parse(data: bytes) -> dict:
    """
    Parsea un mensaje protobuf de forma minimal.
    Devuelve {field_number: [values]}
      - wire_type 0 (varint): value = int
      - wire_type 2 (len-delimited): value = bytes
    """
    s = io.BytesIO(data)
    n = len(data)
    out: dict = {}
    while s.tell() < n:
        tag = _varint(s)
        if tag is None:
            break
        fn, wt = tag >> 3, tag & 0x7
        if wt == 0:
            v = _varint(s)
            if v is not None:
                out.setdefault(fn, []).append(v)
        elif wt == 1:
            s.read(8)
        elif wt == 2:
            length = _varint(s)
            if length is not None:
                out.setdefault(fn, []).append(s.read(length))
        elif wt == 5:
            s.read(4)
        else:
            break  # wire type desconocido — detenerse
    return out


def _varints(data: bytes):
    """Generador: lee todos los varints de un campo packed."""
    s = io.BytesIO(data)
    n = len(data)
    while s.tell() < n:
        v = _varint(s)
        if v is None:
            break
        yield v


def _zigzag(v: int) -> int:
    """Decodifica sint64 zigzag → int con signo."""
    return (v >> 1) ^ -(v & 1)


# ── Procesamiento de bloques ──────────────────────────────────────────────────

def _process_block(block_bytes: bytes,
                   minlat: float, minlon: float,
                   maxlat: float, maxlon: float,
                   records: list) -> None:
    block = _parse(block_bytes)

    # StringTable: campo 1, sub-mensaje con campo 1 repetido (bytes)
    stringtable: list[str] = []
    if 1 in block:
        st = _parse(block[1][0])
        stringtable = [s.decode("utf-8", errors="replace") for s in st.get(1, [])]

    if not stringtable:
        return

    try:
        hn_key = stringtable.index("addr:housenumber")
    except ValueError:
        return  # ningún addr:housenumber en este bloque

    granularity = block.get(17, [100])[0]   # nanogrados por unidad (default 100)
    lat_off     = block.get(19, [0])[0]     # offset latitud (nanogrados)
    lon_off     = block.get(20, [0])[0]     # offset longitud (nanogrados)

    scale = 1e-9  # nanogrados → grados

    for pg_bytes in block.get(2, []):
        pg = _parse(pg_bytes)

        # ── DenseNodes (campo 2) ──────────────────────────────────────────────
        for dn_bytes in pg.get(2, []):
            dn = _parse(dn_bytes)

            if 8 not in dn or 9 not in dn:
                continue

            # Latitudes y longitudes: delta-coded sint64 packed
            lats_raw = list(_varints(dn[8][0]))
            lons_raw = list(_varints(dn[9][0]))
            n_nodes = len(lats_raw)

            # Delta-decode + zigzag + escala
            act_lats: list[float] = []
            act_lons: list[float] = []
            cl = cl2 = 0
            for i in range(n_nodes):
                cl  += _zigzag(lats_raw[i])
                cl2 += _zigzag(lons_raw[i])
                act_lats.append((lat_off + granularity * cl)  * scale)
                act_lons.append((lon_off + granularity * cl2) * scale)

            # keys_vals (campo 10): pares key/val separados por 0
            kv_data = dn.get(10, [b""])[0]
            node_tags: list[dict] = []
            cur: dict = {}
            kv_iter = _varints(kv_data)
            for k in kv_iter:
                if k == 0:
                    node_tags.append(cur)
                    cur = {}
                else:
                    v = next(kv_iter, None)
                    if (v is not None
                            and k < len(stringtable)
                            and v < len(stringtable)):
                        cur[stringtable[k]] = stringtable[v]
            # Nodos restantes sin separador final explícito
            if cur:
                node_tags.append(cur)

            for i in range(n_nodes):
                if i >= len(node_tags):
                    break
                tags = node_tags[i]
                if "addr:housenumber" not in tags:
                    continue
                lat, lon = act_lats[i], act_lons[i]
                if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
                    continue
                _push(tags["addr:housenumber"], lat, lon, records)

        # ── Regular Nodes (campo 1, menos común en PBF moderno) ──────────────
        for node_bytes in pg.get(1, []):
            node = _parse(node_bytes)
            if 2 not in node or 3 not in node:
                continue
            keys = list(_varints(node[2][0]))
            if hn_key not in keys:
                continue
            if 8 not in node or 9 not in node:
                continue
            lat = (lat_off + granularity * _zigzag(node[8][0])) * scale
            lon = (lon_off + granularity * _zigzag(node[9][0])) * scale
            if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
                continue
            vals = list(_varints(node[3][0]))
            idx = keys.index(hn_key)
            if idx < len(vals) and vals[idx] < len(stringtable):
                _push(stringtable[vals[idx]], lat, lon, records)


def _push(raw: str, lat: float, lon: float, records: list) -> None:
    """Normaliza el housenumber y lo agrega a records si es válido."""
    token = raw.strip().split()[0] if raw.strip() else ""
    num_str = "".join(c for c in token if c.isdigit())
    if not num_str:
        return
    try:
        number = int(num_str)
    except ValueError:
        return
    if 1 <= number <= 99999:
        records.append({"number": number, "lat": lat, "lon": lon})


# ── Tipos de vía incluidos en la red "drive" ─────────────────────────────────

HIGHWAY_DRIVE = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link",
    "living_street", "service", "road",
}


def _collect_nodes(pbf_path: str,
                   minlat: float, minlon: float,
                   maxlat: float, maxlon: float) -> dict:
    """
    Pasada 1: devuelve {node_id: (lat, lon)} para todos los nodos
    dentro del bbox expandido.
    """
    nodes: dict = {}
    file_size = os.path.getsize(pbf_path)
    blob_count = 0

    with open(pbf_path, "rb") as f:
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            header_len = struct.unpack(">I", raw)[0]
            header = _parse(f.read(header_len))
            blob_type = header[1][0].decode("utf-8") if 1 in header else ""
            data_size = header.get(3, [0])[0]
            blob = _parse(f.read(data_size))

            if 1 in blob:
                raw_data = blob[1][0]
            elif 3 in blob:
                raw_data = zlib.decompress(blob[3][0])
            else:
                continue

            if blob_type != "OSMData":
                continue

            blob_count += 1
            if blob_count % 100 == 0:
                pct = f.tell() / file_size * 100
                print(f"\r      [1/2] {pct:5.1f}%  —  {len(nodes):,} nodos en bbox",
                      end="", flush=True)

            block = _parse(raw_data)
            granularity = block.get(17, [100])[0]
            lat_off = block.get(19, [0])[0]
            lon_off = block.get(20, [0])[0]
            scale = 1e-9

            for pg_bytes in block.get(2, []):
                pg = _parse(pg_bytes)

                # DenseNodes
                for dn_bytes in pg.get(2, []):
                    dn = _parse(dn_bytes)
                    if 1 not in dn or 8 not in dn or 9 not in dn:
                        continue
                    ids_raw  = list(_varints(dn[1][0]))
                    lats_raw = list(_varints(dn[8][0]))
                    lons_raw = list(_varints(dn[9][0]))
                    n = min(len(ids_raw), len(lats_raw), len(lons_raw))
                    cid = clat = clon = 0
                    for i in range(n):
                        cid  += _zigzag(ids_raw[i])
                        clat += _zigzag(lats_raw[i])
                        clon += _zigzag(lons_raw[i])
                        lat = (lat_off + granularity * clat) * scale
                        lon = (lon_off + granularity * clon) * scale
                        if minlat <= lat <= maxlat and minlon <= lon <= maxlon:
                            nodes[cid] = (lat, lon)

    print()
    return nodes


def _collect_ways(pbf_path: str, nodes: dict, highway_types: set) -> list:
    """
    Pasada 2: lee todos los ways con highway en highway_types y
    cuyos nodos estén (al menos parcialmente) en el dict de nodos.
    Devuelve lista de {'name', 'highway', 'coords': [(lat, lon), ...]}.
    """
    ways: list = []
    file_size = os.path.getsize(pbf_path)
    blob_count = 0

    with open(pbf_path, "rb") as f:
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            header_len = struct.unpack(">I", raw)[0]
            header = _parse(f.read(header_len))
            blob_type = header[1][0].decode("utf-8") if 1 in header else ""
            data_size = header.get(3, [0])[0]
            blob = _parse(f.read(data_size))

            if 1 in blob:
                raw_data = blob[1][0]
            elif 3 in blob:
                raw_data = zlib.decompress(blob[3][0])
            else:
                continue

            if blob_type != "OSMData":
                continue

            blob_count += 1
            if blob_count % 100 == 0:
                pct = f.tell() / file_size * 100
                print(f"\r      [2/2] {pct:5.1f}%  —  {len(ways):,} segmentos",
                      end="", flush=True)

            block = _parse(raw_data)
            stringtable: list[str] = []
            if 1 in block:
                st = _parse(block[1][0])
                stringtable = [s.decode("utf-8", errors="replace")
                               for s in st.get(1, [])]
            if not stringtable:
                continue

            for pg_bytes in block.get(2, []):
                pg = _parse(pg_bytes)

                for way_bytes in pg.get(3, []):
                    way = _parse(way_bytes)
                    keys = list(_varints(way[2][0])) if 2 in way else []
                    vals = list(_varints(way[3][0])) if 3 in way else []

                    tags: dict = {}
                    for k, v in zip(keys, vals):
                        if k < len(stringtable) and v < len(stringtable):
                            tags[stringtable[k]] = stringtable[v]

                    hw = tags.get("highway", "")
                    if hw not in highway_types:
                        continue

                    refs_raw = list(_varints(way[8][0])) if 8 in way else []
                    ref_id = 0
                    coords = []
                    for r in refs_raw:
                        ref_id += _zigzag(r)
                        if ref_id in nodes:
                            coords.append(nodes[ref_id])

                    if len(coords) < 2:
                        continue

                    ways.append({
                        "name":    tags.get("name", ""),
                        "highway": hw,
                        "coords":  coords,
                    })

    print()
    return ways


def extract_street_network(pbf_path: str, bbox: dict,
                           highway_types: set = None) -> "gpd.GeoDataFrame":
    """
    Lee el PBF en dos pasadas y devuelve un GeoDataFrame con los segmentos
    de la red vial dentro del bbox. No requiere conexión a internet.

    bbox: dict con claves minlon, minlat, maxlon, maxlat
    """
    from shapely.geometry import LineString
    import geopandas as gpd

    if highway_types is None:
        highway_types = HIGHWAY_DRIVE

    pad = 0.01  # ~1 km extra alrededor del bbox para capturar nodos de borde

    nodes = _collect_nodes(
        pbf_path,
        minlat=bbox["minlat"] - pad,
        minlon=bbox["minlon"] - pad,
        maxlat=bbox["maxlat"] + pad,
        maxlon=bbox["maxlon"] + pad,
    )
    print(f"      {len(nodes):,} nodos recopilados.")

    ways = _collect_ways(pbf_path, nodes, highway_types)
    print(f"      {len(ways):,} segmentos de calle encontrados.")

    records = []
    for w in ways:
        # Shapely usa (x=lon, y=lat)
        coords_xy = [(lon, lat) for lat, lon in w["coords"]]
        records.append({
            "geometry": LineString(coords_xy),
            "name":     w["name"],
            "highway":  w["highway"],
        })

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ── API pública ───────────────────────────────────────────────────────────────

def extract_addresses(pbf_path: str, bbox: dict) -> list:
    """
    Lee un archivo OSM PBF y extrae todos los nodos con addr:housenumber
    dentro del bounding box indicado.

    bbox: dict con claves minlon, minlat, maxlon, maxlat

    Devuelve: list de {'number': int, 'lat': float, 'lon': float}
    """
    minlon = bbox["minlon"]
    minlat = bbox["minlat"]
    maxlon = bbox["maxlon"]
    maxlat = bbox["maxlat"]

    records: list = []
    file_size = os.path.getsize(pbf_path)
    blob_count = 0

    with open(pbf_path, "rb") as f:
        while True:
            # ── Leer BlobHeader ───────────────────────────────────────────────
            raw = f.read(4)
            if len(raw) < 4:
                break
            header_len = struct.unpack(">I", raw)[0]
            header = _parse(f.read(header_len))

            blob_type = (header[1][0].decode("utf-8")
                         if 1 in header else "")
            data_size = header.get(3, [0])[0]

            # ── Leer Blob ─────────────────────────────────────────────────────
            blob = _parse(f.read(data_size))

            if 1 in blob:                  # raw (sin compresión)
                raw_data = blob[1][0]
            elif 3 in blob:                # zlib_data
                raw_data = zlib.decompress(blob[3][0])
            else:
                continue

            if blob_type == "OSMData":
                blob_count += 1
                if blob_count % 100 == 0:
                    pct = f.tell() / file_size * 100
                    print(
                        f"\r      {pct:5.1f}%  —  {len(records):,} puntos encontrados",
                        end="", flush=True
                    )
                _process_block(raw_data, minlat, minlon, maxlat, maxlon, records)

    print()  # salto de línea tras el progreso
    return records
