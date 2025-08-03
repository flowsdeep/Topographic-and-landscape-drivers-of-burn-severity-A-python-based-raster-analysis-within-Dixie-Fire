# Topographic and landscape drivers of burn severity:
A python-based raster analysis of one watershed within the Dixie Fire

**Author**: Sam Freedman  
**Course**: GEOG 721 (Raster GIS), San Francisco State University

---

## Project Overview

Using linear regression and nonlinear Random Forest models, this project explores how topographic and landscape features influence burn severity (measured via RdNBR) in the **Mosquito Creek–North Fork Feather River watershed**, located within the 2021 **Dixie Fire** perimeter in Northern California.

### Goals:
- Identify key predictors of burn severity using terrain and anthropogenic variables
- Integrate spatial data including DEM-derived metrics, roads, and LANDFIRE vegetation layers
- Evaluate residual spatial structure
- Apply spatial cross-validation and stratified modeling by vegetation type

---

## Data Sources

- **Burn Severity (RdNBR)**: MTBS (Monitoring Trends in Burn Severity)
- **Topographic Variables**: Derived from USGS 10m DEM
  - Slope, Elevation, TPI, Solar Radiation
- **Roads**: TIGER/Line Primary and Forest Service Roads
- **Vegetation / Fuel Metrics**: LANDFIRE EVT, Canopy Cover, Canopy Bulk Density
- **Watershed Boundary**: NHD HU-12

---

## Tools & Technologies

- `Python 3.9+`
- `ArcPy` (via ArcGIS Pro)
- `GeoPandas`, `pandas`, `numpy`
- `scikit-learn`, `statsmodels`
- `matplotlib`, `seaborn`
- `Jupyter Notebooks` (recommended)

---

## Methodology Overview

1. **Preprocessing**  
   - DEM preprocessing and variable derivation (Slope, TPI, etc.)
   - Clipping LANDFIRE rasters to the watershed
   - Spatial joins of raster values to random points

2. **Regression Modeling**  
   - OLS for initial model and residual extraction  
   - Random Forest to capture nonlinear relationships  
   - Feature importance ranking

3. **Spatial Validation**  
   - Moran’s I for spatial autocorrelation of residuals  
   - Spatially-blocked cross-validation using GroupKFold  

4. **Vegetation-Type Analysis**  
   - RdNBR summarized by LANDFIRE EVT class  
   - RF models stratified by vegetation type  
   - Binned burn severity classification (Low/Med/High)

---

## 📂 File Structure

