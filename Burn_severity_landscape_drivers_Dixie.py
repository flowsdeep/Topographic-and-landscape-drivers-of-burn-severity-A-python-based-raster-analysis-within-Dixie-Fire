# Importing data (DEM, forest roads, primary roads, RdNBR, watersheds)

import arcpy
import os

arcpy.env.overwriteOutput = True

project = r"C:\Users\patron\Desktop\721 GIS\Final Project\721_Final_Project"
arcpy.env.workspace = project

forest_roads = os.path.join(project, "Roads", "S_USA_RoadCore_FS.shp")
primary_roads = os.path.join(project, "Roads", "tl_2024_06_prisecroads.shp")
rdnbr = os.path.join(project, "MTBS", "ca3987612137920210714_20200708_20220714_rdnbr.tif")
watersheds = os.path.join(project, "NHD_H_18020121_HU8_GDB.gdb", "WBD", "WBDHU12")
dem = os.path.join(project, "us_orig_dem", "orig_dem")

aprx = arcpy.mp.ArcGISProject("CURRENT")
mapx = aprx.listMaps()[0]

mapx.addDataFromPath(forest_roads)
mapx.addDataFromPath(primary_roads)
mapx.addDataFromPath(rdnbr)
mapx.addDataFromPath(watersheds)
mapx.addDataFromPath(dem)



# Choosing a representative watershed

wbd_path = os.path.join(project, "NHD_H_18020121_HU8_GDB.gdb", "WBD", "WBDHU12")
output_gdb = os.path.join(project, "721_Final_Project.gdb")
output_fc = os.path.join(output_gdb, "MosquitoCreek_NorthFork_FeatherRiver")

where_clause = "NAME = 'Mosquito Creek-North Fork Feather River'"
layer_name = "WBDHU12_lyr"

arcpy.MakeFeatureLayer_management(wbd_path, layer_name)
arcpy.SelectLayerByAttribute_management(layer_name, "NEW_SELECTION", where_clause)
arcpy.CopyFeatures_management(layer_name, output_fc)



# Clipping features (DEM, roads, & RdNBR) to watershed

from arcpy.sa import ExtractByMask

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

gdb = os.path.join(project, "721_Final_Project.gdb")

forest_roads = os.path.join(project, "Roads", "S_USA_RoadCore_FS.shp")
primary_roads = os.path.join(project, "Roads", "tl_2024_06_prisecroads.shp")
dem = os.path.join(project, "us_orig_dem", "orig_dem")
rdnbr = os.path.join(project, "MTBS", "ca3987612137920210714_20200708_20220714_rdnbr.tif")
clip_feature = os.path.join(gdb, "MosquitoCreek_NorthFork_FeatherRiver")

target_sr = arcpy.SpatialReference(26910)

projected_watershed = os.path.join(gdb, "Watershed_UTM10N")
arcpy.management.Project(clip_feature, projected_watershed, target_sr)

roads_projected = os.path.join(gdb, "Forest_Roads_Projected")
primary_roads_projected = os.path.join(gdb, "Primary_Roads_Projected")
arcpy.management.Project(forest_roads, roads_projected, target_sr)
arcpy.management.Project(primary_roads, primary_roads_projected, target_sr)

clipped_forest_roads = os.path.join(gdb, "Clipped_Forest_Roads")
clipped_primary_roads = os.path.join(gdb, "Clipped_Primary_Roads")
arcpy.analysis.Clip(roads_projected, projected_watershed, clipped_forest_roads)
arcpy.analysis.Clip(primary_roads_projected, projected_watershed, clipped_primary_roads)

clipped_dem = os.path.join(gdb, "Clipped_DEM")
clipped_rdnbr = os.path.join(gdb, "Clipped_RdNBR")
ExtractByMask(dem, projected_watershed).save(clipped_dem)
ExtractByMask(rdnbr, projected_watershed).save(clipped_rdnbr)


# Reprojecting rasters to NAD83 UTM Zone 10N

import arcpy
import os

arcpy.env.overwriteOutput = True

project_dir = os.getcwd()
gdb = os.path.join(project_dir, "721_Final Project.gdb")
reproj_dir = os.path.join(project_dir, "Reprojected")
os.makedirs(reproj_dir, exist_ok=True)

clipped_dem = os.path.join(gdb, "Clipped_DEM")
clipped_rdnbr = os.path.join(gdb, "Clipped_RdNBR")

dem_out = os.path.join(reproj_dir, "DEM_UTM10.tif")
rdnbr_out = os.path.join(reproj_dir, "RdNBR_UTM10.tif")

target_sr = arcpy.SpatialReference(26910)

arcpy.management.ProjectRaster(
    in_raster=clipped_dem,
    out_raster=dem_out,
    out_coor_system=target_sr,
    resampling_type="BILINEAR"
)

arcpy.management.ProjectRaster(
    in_raster=clipped_rdnbr,
    out_raster=rdnbr_out,
    out_coor_system=target_sr,
    resampling_type="BILINEAR"
)


# Deriving slope and TPI from DEM

from arcpy.sa import *

arcpy.env.snapRaster = os.path.join(reproj_dir, "DEM_UTM10.tif")

dem = os.path.join(reproj_dir, "DEM_UTM10.tif")
slope_path = os.path.join(gdb, "slope_utm")
tpi_path = os.path.join(gdb, "tpi_utm")

for path in [slope_path, tpi_path]:
    if arcpy.Exists(path):
        arcpy.management.Delete(path)

slope = Slope(dem, output_measurement="DEGREE")
slope.save(slope_path)

focal_mean = FocalStatistics(dem, NbrCircle(5, "CELL"), "MEAN", "DATA")
tpi = Raster(dem) - focal_mean
tpi.save(tpi_path)


# Deriving distance from forest roads

from arcpy.sa import *

roads_fc = os.path.join(gdb, "Clipped_Forest_Roads")
euclidean_output = os.path.join(gdb, "Dist_Roads")

if arcpy.Exists(euclidean_output):
    arcpy.management.Delete(euclidean_output)

euc_distance = EucDistance(roads_fc)
euc_distance.save(euclidean_output)


# Creating random points and extracting multivalues to points

arcpy.management.Delete("Sample_RandomPoints")

watershed = os.path.join(gdb, "Watershed_UTM10N")
random_points = os.path.join(gdb, "Sample_RandomPoints")

arcpy.management.CreateRandomPoints(
    out_path=gdb,
    out_name="Sample_RandomPoints",
    constraining_feature_class=watershed,
    number_of_points_or_field=1000
)

raster_inputs = [
    (os.path.join(gdb, "slope_utm"), "Slope"),
    (os.path.join(reproj_dir, "DEM_UTM10.tif"), "Elevation"),
    (os.path.join(gdb, "tpi_utm"), "TPI"),
    (os.path.join(gdb, "solarradiation"), "Solar"),
    (os.path.join(gdb, "Dist_Roads"), "DistRoads"),
    (os.path.join(reproj_dir, "RdNBR_UTM10.tif"), "RdNBR")
]

arcpy.env.snapRaster = os.path.join(reproj_dir, "DEM_UTM10.tif")

ExtractMultiValuesToPoints(random_points, raster_inputs, "BILINEAR")


# Converting to DataFrame & exploring Pearson correlation

import pandas as pd
import numpy as np
import arcpy

gdb = os.path.join(project, "721_Final Project.gdb")
points = os.path.join(gdb, "Sample_RandomPoints")

fields = ["Slope", "Elevation", "TPI", "Solar", "DistRoads", "RdNBR"]

arr = arcpy.da.FeatureClassToNumPyArray(points, fields)
df = pd.DataFrame(arr)
df_clean = df.dropna()

print(df_clean.describe())


# Ordinary Least Squares regression to investigate further

import statsmodels.api as sm

df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df_clean[["Slope", "Elevation", "TPI", "Solar", "DistRoads"]]
X = sm.add_constant(X)
y = df_clean["RdNBR"]

model = sm.OLS(y, X).fit()
print(model.summary())


# Checking for multicollinearity (VIF)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

# Residual mapping and joining to points
import arcpy
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import os

project = r"C:\Users\patron\Desktop\721 GIS\Final Project\721_Final_Project"
gdb = os.path.join(project, "721_Final_Project.gdb")
points_fc = os.path.join(gdb, "Sample_RandomPoints")
output_path = os.path.join(project, "RandomPoints_Residuals.shp")

fields = ["OID@", "Slope", "Elevation", "TPI", "Solar", "DistRoads", "RdNBR"]
arr = arcpy.da.FeatureClassToNumPyArray(points_fc, fields)
df = pd.DataFrame(arr)

df_clean = df.dropna().copy()
X = df_clean[["Slope", "Elevation", "TPI", "Solar", "DistRoads"]]
X = sm.add_constant(X)
y = df_clean["RdNBR"]
model = sm.OLS(y, X).fit()
df_clean["residuals"] = model.resid

gdf = gpd.read_file(gdb, layer="Sample_RandomPoints")
df_clean["OID@"] = df_clean["OID@"].astype(int)
gdf["OID@"] = gdf.index.astype(int)
gdf_merged = gdf.merge(df_clean[["OID@", "residuals"]], on="OID@", how="left")
gdf_merged.to_file(output_path)
print(output_path)


# Random Forest (non-parametric) after Moran’s I showed low spatial autocorrelation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

X = df_clean[["Slope", "Elevation", "TPI", "Solar", "DistRoads"]]
y = df_clean["RdNBR"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test R²: {r2:.3f}")
print(f"Test RMSE: {rmse:.2f}")

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.plot(kind="bar", color="forestgreen")
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
plt.xlabel("Actual RdNBR")
plt.ylabel("Predicted RdNBR")
plt.title("Random Forest: Actual vs Predicted (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Spatial cross-validation

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

gdf = gpd.read_file("721_Final_Project/RandomPoints_Residuals.shp")

features = ["Slope", "Elevation", "TPI", "Solar", "DistRoads"]
target = "RdNBR"

gdf["spatial_block"] = (gdf.geometry.x // 1000).astype(int).astype(str) + "_" + (gdf.geometry.y // 1000).astype(int).astype(str)
X = gdf[features]
y = gdf[target]
groups = gdf["spatial_block"]

cv = GroupKFold(n_splits=5)
r2_scores = []
rmse_scores = []

for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

print(f"Average R² (spatial CV): {np.mean(r2_scores):.3f}")
print(f"Average RMSE (spatial CV): {np.mean(rmse_scores):.2f}")


# Importing vegetation data from LANDFIRE (EVT, canopy cover, CBD) and extracting to sample points

from arcpy.sa import ExtractByMask

veg_dir = os.path.join(project, "LF2020_EVT_220_CONUS")
evt = os.path.join(veg_dir, "LC20_EVT_220.tif")
cc = os.path.join(veg_dir, "LC20_CC_220.tif")
cbd = os.path.join(veg_dir, "LC20_CBD_220.tif")

watershed = os.path.join(gdb, "Watershed_UTM10N")
sample_points = os.path.join(gdb, "Sample_RandomPoints")

clipped_evt = os.path.join(gdb, "Clipped_VegType.tif")
clipped_cc = os.path.join(gdb, "Clipped_CanopyCover.tif")
clipped_cbd = os.path.join(gdb, "Clipped_CanopyBulkDensity.tif")

proj_evt = os.path.join(gdb, "Projected_VegType.tif")
proj_cc = os.path.join(gdb, "Projected_CanopyCover.tif")
proj_cbd = os.path.join(gdb, "Projected_CanopyBulkDensity.tif")

target_sr = arcpy.SpatialReference(26910)

arcpy.CheckOutExtension("Spatial")
ExtractByMask(evt, watershed).save(clipped_evt)
ExtractByMask(cc, watershed).save(clipped_cc)
ExtractByMask(cbd, watershed).save(clipped_cbd)

arcpy.management.ProjectRaster(clipped_evt, proj_evt, target_sr, "NEAREST")
arcpy.management.ProjectRaster(clipped_cc, proj_cc, target_sr, "BILINEAR")
arcpy.management.ProjectRaster(clipped_cbd, proj_cbd, target_sr, "BILINEAR")

arcpy.sa.ExtractMultiValuesToPoints(
    sample_points,
    [[proj_evt, "VegType"], [proj_cc, "CanopyCover"], [proj_cbd, "CBD"]],
    "BILINEAR"
)


# Preparing data for updated Random Forest

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gdb = os.path.join(project, "721_Final_Project.gdb")
fc_path = os.path.join(gdb, "Sample_RandomPoints")

fields = [
    "Slope", "Elevation", "TPI", "Solar", "DistRoads",
    "VegType", "CanopyCover", "CBD", "RdNBR"
]

data = [row for row in arcpy.da.SearchCursor(fc_path, fields)]
df = pd.DataFrame(data, columns=fields)
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

features = ["Slope", "Elevation", "TPI", "Solar", "DistRoads", "VegType", "CanopyCover", "CBD"]
X = df_clean[features]
y = df_clean["RdNBR"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar([features[i] for i in indices], importances[indices], color="forestgreen")
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
plt.xlabel("Actual RdNBR")
plt.ylabel("Predicted RdNBR")
plt.title("RF: Actual vs Predicted RdNBR")
plt.grid(True)
plt.tight_layout()
plt.show()


# Groupby mean RdNBR by vegetation type

import seaborn as sns

gdf = gpd.read_file(fc_path)
gdf["VegType"] = gdf["VegType"].astype("category")
veg_summary = gdf.groupby("VegType")["RdNBR"].agg(["mean", "count"]).reset_index()
veg_summary = veg_summary[veg_summary["count"] >= 30]

def classify_severity(rdnbr):
    if rdnbr < 100:
        return "Low"
    elif rdnbr < 400:
        return "Medium"
    else:
        return "High"

veg_summary["SeverityClass"] = veg_summary["mean"].apply(classify_severity)
veg_summary = veg_summary.sort_values(by="mean", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=veg_summary, x="VegType", y="mean", hue="SeverityClass",
            palette={"Low": "limegreen", "Medium": "orange", "High": "red"}, dodge=False)
plt.axhline(100, color="gray", linestyle="--", label="Low/Medium threshold")
plt.axhline(400, color="gray", linestyle="--", label="Medium/High threshold")
plt.xlabel("Vegetation Type Code (EVT)")
plt.ylabel("Mean RdNBR")
plt.title("Mean RdNBR by Vegetation Type")
plt.legend(title="Severity Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Random Forest stratified by vegetation type

rf_results = []

for veg in gdf["VegType"].cat.dropna().unique():
    subset = gdf[gdf["VegType"] == veg]
    if len(subset) < 30:
        continue  # Skip under-sampled vegetation types

    X = subset[features]
    y = subset["RdNBR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rf_results.append({
        "VegType": veg,
        "n": len(subset),
        "R²": round(r2_score(y_test, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    })

rf_df = pd.DataFrame(rf_results).sort_values(by="R²", ascending=False)
print("\nRandom Forest by Vegetation Type")
print(rf_df)



