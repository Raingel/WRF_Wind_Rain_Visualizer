#!/usr/bin/env python3
# coding: utf-8

import os
import re
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour


# =========================
# Config
# =========================
WRF_DATALIST = [
    "M-A0064-000", "M-A0064-006", "M-A0064-012", "M-A0064-018", "M-A0064-024",
    "M-A0064-030", "M-A0064-036", "M-A0064-042", "M-A0064-048", "M-A0064-054",
    "M-A0064-060", "M-A0064-066", "M-A0064-072", "M-A0064-078", "M-A0064-084",
]

MATCH_REGEX = r"TMP:2 m|NSWRS:surface|UGRD:10 m|VGRD:10 m|RH:2 m|APCP:surface"

# Taiwan bbox (跟你原本一致)
LON_MIN, LON_MAX = 119.9, 122.0
LAT_MIN, LAT_MAX = 21.9, 25.5

# interpolation 網格密度（跟你原本一致）
APCP_GRID_NX, APCP_GRID_NY = 200, 200
WIND_GRID_NX, WIND_GRID_NY = 10, 50

# contour levels（跟你原本一致）
CONTOUR_LEVELS = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 9999]

COLORS = [
    "#ffffff", "#9cfeff", "#01d2fd", "#00a6fc", "#0177fd", "#27a41c", "#01f92f",
    "#fffe32", "#ffd328", "#ffa81b", "#ff2b06", "#da2203", "#aa1703", "#aa1fa2",
    "#dc2dd2", "#ff37fa", "#ffd6fe"
]


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("wrf_pipeline")


# =========================
# Paths
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
GRIB_DIR = DATA_DIR / "grib"
CSV_DIR = DATA_DIR / "csv"
LATEST_CSV_GZ = DATA_DIR / "wrf_latest.csv.gz"

OUT_DIR = REPO_ROOT / "docs" / "wrf_tmp"
DATE_RANGE_JSON = OUT_DIR / "wrf_date_range.json"


def ensure_dirs():
    GRIB_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str:
    key = os.environ.get("CWA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing env var CWA_API_KEY. Please set it in GitHub Secrets and pass to workflow.")
    return key


def robust_get(session: requests.Session, url: str, timeout=60, retries=5) -> requests.Response:
    last_err = None
    for i in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            logger.warning("GET failed (%s/%s): %s", i + 1, retries, str(e))
    raise RuntimeError(f"GET failed after {retries} retries: {url}\nLast error: {last_err}")


def extract_grb_uri(meta_json: dict) -> str:
    # 你原本用的路徑
    try:
        return meta_json["cwaopendata"]["dataset"]["resource"]["uri"]
    except Exception:
        pass

    # 防守性：有些 open data 可能是 resources list
    try:
        res = meta_json["cwaopendata"]["dataset"]["resources"]["resource"]
        if isinstance(res, list) and len(res) > 0:
            return res[0]["uri"]
        if isinstance(res, dict) and "uri" in res:
            return res["uri"]
    except Exception:
        pass

    raise KeyError("Cannot locate GRB URI in metadata JSON.")


def download_grib2_files(api_key: str):
    session = requests.Session()
    for ds in WRF_DATALIST:
        meta_url = f"https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/{ds}?Authorization={api_key}&downloadType=WEB&format=JSON"
        logger.info("Fetch meta: %s", ds)
        meta_r = robust_get(session, meta_url, timeout=60, retries=5)
        meta = meta_r.json()

        grb_uri = extract_grb_uri(meta)
        logger.info("GRB URI: %s", grb_uri)

        out_grb = GRIB_DIR / f"{ds}.grb2"
        logger.info("Download GRIB2: %s -> %s", ds, out_grb)
        grb_r = robust_get(session, grb_uri, timeout=180, retries=8)
        out_grb.write_bytes(grb_r.content)


def run_wgrib2_to_csv():
    # wgrib2 由 conda-forge 裝好後，binary 會在 PATH
    for ds in WRF_DATALIST:
        grb_path = GRIB_DIR / f"{ds}.grb2"
        if not grb_path.exists():
            logger.warning("Missing GRIB2: %s (skip)", grb_path)
            continue

        csv_path = CSV_DIR / f"{ds}.csv"
        logger.info("wgrib2 -> csv: %s", ds)

        cmd = [
            "wgrib2",
            str(grb_path),
            "-match", MATCH_REGEX,
            "-csv", str(csv_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("wgrib2 failed: %s\nstdout:\n%s\nstderr:\n%s", ds, proc.stdout, proc.stderr)
            raise RuntimeError(f"wgrib2 failed for {ds}")


def normalize_var_name(s: str) -> str:
    # "TMP:2 m" -> "TMP"
    return str(s).split(":")[0].strip()


def parse_time_like(x: str):
    """
    盡量吃下 wgrib2 常見的時間格式
    可能有:
      2026-02-04 00:00:00
      2026020400
      d=2026020400
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    s = s.replace("d=", "")
    # 10 digits: YYYYMMDDHH
    if re.fullmatch(r"\d{10}", s):
        return pd.to_datetime(s, format="%Y%m%d%H", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def build_latest_wrf_table() -> pd.DataFrame:
    dfs = []
    for ds in WRF_DATALIST:
        csv_path = CSV_DIR / f"{ds}.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as e:
            logger.warning("Read csv failed: %s (%s)", csv_path, str(e))
            continue

        # 依照你原始程式的欄位假設：0 Initial, 1 Valid, 2 Var, 4 Lon, 5 Lat, 6 Value
        # 如果 wgrib2 格式有額外欄位，只要至少到 6 就能用
        if df.shape[1] < 7:
            logger.warning("Unexpected csv columns: %s cols=%s (skip)", csv_path, df.shape[1])
            continue

        df = df[[0, 1, 2, 4, 5, 6]].copy()
        df.columns = ["Initial", "Valid", "Var", "Lon", "Lat", "Value"]

        df["Var"] = df["Var"].map(normalize_var_name)
        df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
        df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        df["Initial"] = df["Initial"].map(parse_time_like)
        df["Valid"] = df["Valid"].map(parse_time_like)

        # Taiwan bbox
        df = df[
            (df["Lon"] > LON_MIN) & (df["Lon"] < LON_MAX) &
            (df["Lat"] > LAT_MIN) & (df["Lat"] < LAT_MAX)
        ].dropna(subset=["Initial", "Valid", "Lon", "Lat", "Var", "Value"])

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSV data loaded. Check download/wgrib2 steps.")

    all_df = pd.concat(dfs, ignore_index=True)
    latest_initial = all_df["Initial"].max()
    latest_df = all_df[all_df["Initial"] == latest_initial].copy()

    # pivot 到寬表：一列是一個 (Initial, Valid, Lon, Lat)
    wide = latest_df.pivot_table(
        index=["Initial", "Valid", "Lon", "Lat"],
        columns="Var",
        values="Value",
        aggfunc="first"
    ).reset_index()

    # 存成 gzip csv，簡單且不需要額外引擎
    wide.to_csv(LATEST_CSV_GZ, index=False, compression="gzip")
    logger.info("Saved latest WRF table: %s (rows=%s)", LATEST_CSV_GZ, len(wide))

    return wide


def fool_griddata(df: pd.DataFrame, x_point_num: int, y_point_num: int):
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    new_x_coord = np.linspace(x_min, x_max, x_point_num)
    new_y_coord = np.linspace(y_min, y_max, y_point_num)

    xx, yy = np.meshgrid(new_x_coord, new_y_coord)

    knew_xy_coord = df[["x", "y"]].values
    knew_values = df["value"].values

    result = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method="cubic")
    return xx, yy, result


def interpolation_to_geojson(df_for_interpolation: pd.DataFrame) -> str:
    xx, yy, result = fool_griddata(df_for_interpolation, APCP_GRID_NX, APCP_GRID_NY)
    result = np.nan_to_num(result)

    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot()
    ax.set_aspect("equal", adjustable="box")

    contourf = ax.contourf(
        xx, yy, result,
        levels=CONTOUR_LEVELS,
        colors=COLORS
    )

    gj = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        ndigits=3,
        stroke_width=0,
        fill_opacity=0.5
    )
    plt.close(fig)
    return gj


def reset_out_dir():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_outputs(wide_df: pd.DataFrame):
    # wide_df 已經是 latest initial
    reset_out_dir()

    # date range json
    vmin = wide_df["Valid"].min()
    vmax = wide_df["Valid"].max()
    with open(DATE_RANGE_JSON, "w", encoding="utf-8") as f:
        json.dump([vmin.strftime("%Y-%m-%d %H:%M:%S"), vmax.strftime("%Y-%m-%d %H:%M:%S")], f)
    logger.info("Saved date range: %s", DATE_RANGE_JSON)

    # 6H steps
    wrf_date_range = pd.date_range(start=vmin, end=vmax, freq="6H").to_list()

    def data_extract(df, start, end):
        return df[(df["Valid"] >= start) & (df["Valid"] <= end)]

    # 預先檢查必要欄位
    needed_cols = {"Lon", "Lat", "Valid", "APCP", "VGRD", "UGRD"}
    missing = needed_cols - set(wide_df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in latest table: {missing}")

    for i in range(0, len(wrf_date_range)):
        for j in range(i + 1, len(wrf_date_range)):
            start_t = wrf_date_range[i]
            end_t = wrf_date_range[j]
            tag = f"{start_t.strftime('%Y%m%d_%H%M%S')}_{end_t.strftime('%Y%m%d_%H%M%S')}"

            data_in_span = data_extract(wide_df, start_t, end_t)

            apcp_path = OUT_DIR / f"{tag}_APCP.geojson"
            wind_path = OUT_DIR / f"{tag}_wind.geojson"

            if data_in_span.empty:
                apcp_path.write_text(json.dumps({}), encoding="utf-8")
                wind_path.write_text(json.dumps({}), encoding="utf-8")
                continue

            # APCP diff
            grp = data_in_span.groupby(["Lon", "Lat"])
            diff = grp.max(numeric_only=True) - grp.min(numeric_only=True)

            if "APCP" in diff.columns:
                lonlat_list = [list(ele) for ele in diff.index.to_list()]
                df_interp = pd.DataFrame({
                    "x": np.array(lonlat_list)[:, 0],
                    "y": np.array(lonlat_list)[:, 1],
                    "value": diff["APCP"].fillna(0).values
                })
                gj = interpolation_to_geojson(df_interp)
                apcp_path.write_text(gj, encoding="utf-8")
            else:
                apcp_path.write_text(json.dumps({}), encoding="utf-8")

            # WIND mean
            mean_uv = data_in_span[["Lon", "Lat", "VGRD", "UGRD"]].groupby(["Lon", "Lat"]).mean(numeric_only=True)
            lonlat_list = [list(ele) for ele in mean_uv.index.to_list()]

            df_v = pd.DataFrame({
                "x": np.array(lonlat_list)[:, 0],
                "y": np.array(lonlat_list)[:, 1],
                "value": mean_uv["VGRD"].fillna(0).values
            })
            xx, yy, wv_result = fool_griddata(df_v, WIND_GRID_NX, WIND_GRID_NY)

            df_u = df_v.copy()
            df_u["value"] = mean_uv["UGRD"].fillna(0).values
            _, _, wu_result = fool_griddata(df_u, WIND_GRID_NX, WIND_GRID_NY)

            wv = np.nan_to_num(wv_result)
            wu = np.nan_to_num(wu_result)

            ws = (np.sqrt((wv ** 2) + (wu ** 2)) * 1.94).astype(int)  # m/s -> knots
            wd = ((270 - np.arctan2(wv, wu) * 180 / np.pi) % 360).astype(int)

            lon_list = np.round(xx[0], 3).tolist()
            lat_list = np.round(yy[:, 0], 3).tolist()
            output = {
                "lon_list": lon_list,
                "lat_list": lat_list,
                "ws": ws.tolist(),
                "wd": wd.tolist()
            }
            wind_path.write_text(json.dumps(output), encoding="utf-8")

    logger.info("GeoJSON generation done. Files in: %s", OUT_DIR)


def main():
    ensure_dirs()
    api_key = get_api_key()

    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Start download GRIB2")
    download_grib2_files(api_key)

    logger.info("Start wgrib2 parse")
    run_wgrib2_to_csv()

    logger.info("Build latest wide table")
    wide_df = build_latest_wrf_table()

    logger.info("Generate outputs")
    generate_outputs(wide_df)

    logger.info("All done")


if __name__ == "__main__":
    main()
