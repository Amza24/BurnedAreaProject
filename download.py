import os
import ee
import geemap
import numpy as np
import rasterio
from PIL import Image

# ==========================================
# 1. INIȚIALIZARE GOOGLE EARTH ENGINE
# ==========================================
# Înlocuiește cu ID-ul proiectului tău Google Cloud dacă diferă
ee.Initialize(project='burned-area-and-severity-map')

# Definire AOI (coordonate GPS Grecia)
min_lon, min_lat, max_lon, max_lat = 22.809076, 38.603594, 22.874823, 38.634501
aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

# ==========================================
# 2. INTEROGARE SENTINEL-2 & CALCUL dNBR
# ==========================================
def get_nbr(start_date, end_date):
    img = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
    )
    # NBR = (NIR - SWIR2) / (NIR + SWIR2) -> B8 si B12
    return img.normalizedDifference(['B8', 'B12'])

print("Procesare date satelitare în Earth Engine...")
nbr_pre = get_nbr('2026-06-15', '2026-07-02')
nbr_post = get_nbr('2026-07-15', '2026-08-01')

# Calcul diferență dNBR
dnbr = nbr_pre.subtract(nbr_post)

# Clasificare pe pragurile standard USGS
classified = (
    ee.Image(0)
    .where(dnbr.lt(-0.10), 1)                                    # High Regrowth
    .where(dnbr.gte(-0.10).And(dnbr.lt(0.10)), 2)               # Unburned
    .where(dnbr.gte(0.10).And(dnbr.lt(0.27)), 3)                # Low Severity
    .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 4)                # Mod-Low Severity
    .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 5)                # Mod-High Severity
    .where(dnbr.gte(0.66), 6)                                   # High Severity
)

# ==========================================
# 3. EXPORT TIF & CONVERSIE AUTOMATĂ PNG
# ==========================================
tif_output = "dNBR_classified.tif"
png_output = "harta_severitate_finala.png"

print("Descărcare fișier raster TIF...")
geemap.ee_export_image(
    classified,
    filename=tif_output,
    scale=10,
    region=aoi
)

def convert_tif_to_png(input_tif, output_png):
    with rasterio.open(input_tif) as src:
        data = src.read(1)

    # Culori RGBA (R, G, B, Transparență) pentru codurile 0 - 6
    palette = np.array([
        [0, 0, 0, 0],          # 0: NoData (Transparent)
        [0, 100, 0, 255],      # 1: High Regrowth (Verde închis)
        [124, 252, 0, 255],    # 2: Unburned (Verde deschis)
        [255, 255, 0, 255],    # 3: Low Severity (Galben)
        [255, 165, 0, 255],    # 4: Mod-Low Severity (Portocaliu)
        [255, 69, 0, 255],     # 5: Mod-High Severity (Roșu-Portocaliu)
        [128, 0, 128, 255]     # 6: High Severity (Violet)
    ], dtype=np.uint8)

    rgba_image = palette[data]
    img = Image.fromarray(rgba_image, mode='RGBA')
    img.save(output_png)
    print(f"Conversie completă! Imaginea PNG a fost salvată în:\n{os.path.abspath(output_png)}")

convert_tif_to_png(tif_output, png_output)