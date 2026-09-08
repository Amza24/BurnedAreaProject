import os
import requests
import zipfile
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, transform_geom
import pandas as pd
from shapely.geometry import box, mapping
from PIL import Image

# ==========================================
# 1. CONFIGURARE & AUTENTIFICARE
# ==========================================
USERNAME = "amzanesa24@gmail.com"
PASSWORD = "Amza.Nesa020804"
CLIENT_ID = "cdse-public"
DOWNLOAD_DIR = "copernicus_fire_data"

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Definire AOI din poligonul extras (EPSG:4326 WGS84)
aoi_wgs84 = box(22.809076, 38.603594, 22.874823, 38.634501)


def get_tokens():
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    res = requests.post(token_url, data={
        "client_id": CLIENT_ID, "username": USERNAME,
        "password": PASSWORD, "grant_type": "password"
    })
    if res.status_code == 200:
        return res.json()["access_token"], res.json()["refresh_token"]
    raise Exception(f"Eroare autentificare CDSE: {res.text}")


def download_s2_product(start_date, end_date, token):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    coords = list(aoi_wgs84.exterior.coords)
    poly_str = ", ".join([f"{c[0]} {c[1]}" for c in coords])
    wkt_poly = f"POLYGON(({poly_str}))"

    filter_query = (
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'S2MSI2A') and "
        f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
        f"ContentDate/Start le {end_date}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_poly}')"
    )

    params = {"$filter": filter_query, "$orderby": "ContentDate/Start desc", "$top": 1}
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(url, params=params, headers=headers)

    if res.status_code != 200:
        raise Exception(f"Eroare căutare API ({res.status_code}): {res.text}")

    products = res.json().get("value", [])

    if not products:
        raise Exception(f"Nu s-au găsit imagini între {start_date} și {end_date} pentru perimetrul selectat.")

    prod = products[0]
    title, prod_id = prod["Name"], prod["Id"]

    safe_folder_name = title if title.endswith(".SAFE") else f"{title}.SAFE"
    extract_path = os.path.join(DOWNLOAD_DIR, safe_folder_name)
    zip_path = os.path.join(DOWNLOAD_DIR, f"{title}.zip")

    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path):
            print(f"\n[1/2] Descărcare produs: {title}...")
            dl_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"

            r_head = requests.get(dl_url, headers=headers, allow_redirects=False)
            download_link = r_head.headers.get("Location", dl_url)
            dl_headers = {} if "Location" in r_head.headers else headers

            with requests.get(download_link, headers=dl_headers, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024

                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(
                                    f"\rDescărcat: {downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)",
                                    end="")
            print("\nDescărcare finalizată!")

        print("[2/2] Extragere arhivă ZIP în curs...")
        abs_download_dir = os.path.abspath(DOWNLOAD_DIR)
        win_long_path = f"\\\\?\\{abs_download_dir}" if os.name == 'nt' else abs_download_dir

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(win_long_path)
        print("Extragere completă!")

    return extract_path


# ==========================================
# 2. SELECȚIE BENZI & DECOPARE PE AOI
# ==========================================
def find_band_path(s2_dir, band_name, resolution):
    target_folder = f"R{resolution}"
    for root, _, files in os.walk(s2_dir):
        if target_folder in root:
            for f in files:
                if f.endswith(f"_{band_name}_{resolution}.jp2") or f.endswith(f"_{band_name}.jp2"):
                    return os.path.join(root, f)
    for root, _, files in os.walk(s2_dir):
        for f in files:
            if f"_{band_name}_" in f or f.endswith(f"_{band_name}.jp2"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"Banda {band_name} ({resolution}) nu a fost găsită în {s2_dir}")


def load_cropped_bands(s2_dir, aoi_wgs84_geom):
    path_b08 = find_band_path(s2_dir, "B08", "10m")
    path_b12 = find_band_path(s2_dir, "B12", "20m")

    try:
        path_scl = find_band_path(s2_dir, "SCL", "20m")
    except FileNotFoundError:
        path_scl = None

    with rasterio.open(path_b08) as src08:
        crs = src08.crs
        # Transformare poligon AOI din WGS84 în proiecția nativă UTM a imaginii
        aoi_utm = [transform_geom('EPSG:4326', crs, mapping(aoi_wgs84_geom))]

        # Decupare exactă pe zona de interes
        b08_crop, transform08 = mask(src08, aoi_utm, crop=True, nodata=0)
        b08 = b08_crop[0].astype('float32') / 10000.0
        profile = src08.profile.copy()
        profile.update({
            "height": b08.shape[0],
            "width": b08.shape[1],
            "transform": transform08
        })

    with rasterio.open(path_b12) as src12:
        b12_crop, transform12 = mask(src12, aoi_utm, crop=True, nodata=0)
        b12_raw = b12_crop[0].astype('float32') / 10000.0

    # Re-eșantionare B12 decupată la rezoluția de 10m a B08
    b12 = np.empty_like(b08)
    reproject(
        source=b12_raw, destination=b12,
        src_transform=transform12, src_crs=crs,
        dst_transform=transform08, dst_crs=crs,
        resampling=Resampling.bilinear
    )

    # Mascare nori (SCL)
    if path_scl:
        with rasterio.open(path_scl) as src_scl:
            scl_crop, transform_scl = mask(src_scl, aoi_utm, crop=True, nodata=0)
            scl_raw = scl_crop[0]

        scl = np.empty_like(b08, dtype=np.uint8)
        reproject(
            source=scl_raw, destination=scl,
            src_transform=transform_scl, src_crs=crs,
            dst_transform=transform08, dst_crs=crs,
            resampling=Resampling.nearest
        )
        cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
        b08[cloud_mask] = np.nan
        b12[cloud_mask] = np.nan

    return b08, b12, profile, crs, transform08


# ==========================================
# 3. EXECUTARE PIPELINE ALINIAT
# ==========================================
access_token, _ = get_tokens()

print("Căutare și descărcare imagini...")
dir_pre = download_s2_product("2026-06-15", "2026-07-02", access_token)
dir_post = download_s2_product("2026-07-15", "2026-08-01", access_token)

print("Procesare și decupare benzi spectrale...")
b08_pre, b12_pre, profile, crs_pre, trans_pre = load_cropped_bands(dir_pre, aoi_wgs84)
b08_post_raw, b12_post_raw, _, crs_post, trans_post = load_cropped_bands(dir_post, aoi_wgs84)

# Aliniere spațială exactă a imaginii post-fire pe grila pre-fire
b08_post = np.empty_like(b08_pre)
reproject(
    source=b08_post_raw, destination=b08_post,
    src_transform=trans_post, src_crs=crs_post,
    dst_transform=trans_pre, dst_crs=crs_pre,
    resampling=Resampling.bilinear
)

b12_post = np.empty_like(b08_pre)
reproject(
    source=b12_post_raw, destination=b12_post,
    src_transform=trans_post, src_crs=crs_post,
    dst_transform=trans_pre, dst_crs=crs_pre,
    resampling=Resampling.bilinear
)

# Calcul NBR și dNBR
nbr_pre = (b08_pre - b12_pre) / (b08_pre + b12_pre + 1e-6)
nbr_post = (b08_post - b12_post) / (b08_post + b12_post + 1e-6)
dnbr = nbr_pre - nbr_post

# Identificare valori nevalide
invalid = (b08_pre <= 0) | (b08_post <= 0) | np.isnan(b08_pre) | np.isnan(b08_post)
dnbr[invalid] = np.nan

# Clasificare USGS
dnbr_class = np.zeros_like(dnbr, dtype=np.uint8)
dnbr_class[dnbr < -0.10] = 1
dnbr_class[(dnbr >= -0.10) & (dnbr < 0.10)] = 2
dnbr_class[(dnbr >= 0.10) & (dnbr < 0.27)] = 3
dnbr_class[(dnbr >= 0.27) & (dnbr < 0.44)] = 4
dnbr_class[(dnbr >= 0.44) & (dnbr < 0.66)] = 5
dnbr_class[dnbr >= 0.66] = 6
dnbr_class[np.isnan(dnbr)] = 0

# Calcul statistici de suprafață (1 pixel 10m x 10m = 0.01 ha)
pixel_area_ha = 0.01
labels = ["NoData", "High Regrowth", "Unburned", "Low Severity", "Mod-Low Severity", "Mod-High Severity",
          "High Severity"]
stats = []

for i in range(1, 7):
    count = np.sum(dnbr_class == i)
    stats.append({"Class Code": i, "Severity Class": labels[i], "Area (ha)": round(count * pixel_area_ha, 2)})

df = pd.DataFrame(stats)
df.to_csv("burn_severity_statistics.csv", index=False)

# Export raster GeoTIFF decupat
profile.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open("dNBR_classified.tif", "w", **profile) as dst:
    dst.write(dnbr_class, 1)

print("\nProcesare API finalizată cu succes!")
print(df.to_string(index=False))

# ==========================================
# 4. EXPORT IMAGINE PNG COLORATĂ
# ==========================================
png_output = "harta_severitate_finala.png"

# Culori RGBA pentru codurile de clasă (0 la 6)
palette = np.array([
    [0, 0, 0, 0],          # 0: NoData (Transparent)
    [0, 100, 0, 255],      # 1: High Regrowth (Verde închis)
    [124, 252, 0, 255],    # 2: Unburned (Verde deschis)
    [255, 255, 0, 255],    # 3: Low Severity (Galben)
    [255, 165, 0, 255],    # 4: Mod-Low Severity (Portocaliu)
    [255, 69, 0, 255],     # 5: Mod-High Severity (Roșu-Portocaliu)
    [128, 0, 128, 255]     # 6: High Severity (Violet)
], dtype=np.uint8)

rgba_image = palette[dnbr_class]
img = Image.fromarray(rgba_image, mode='RGBA')
img.save(png_output)

print(f"\nImaginea PNG a fost salvată în: {os.path.abspath(png_output)}")