# evalscripts.py — complete file
# Evalscript snippets used by processors & catalog (Sentinel-Hub PROCESS evalscripts)
# Keep these strings exactly as required by SH (VERSION header, JS functions).

EVAL_S2_TIFF = """\
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B08","B8A","B11"], units: "REFLECTANCE" }],
    output: { bands: 6, sampleType: "FLOAT32" }
  }
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.B8A, s.B11];
}
"""

EVAL_S2_NDVI = """\
//VERSION=3
function setup() {
  return {
    input: [
      { bands: ["B04","B08"], units: "REFLECTANCE" },
      { bands: ["SCL"], units: "DN" }
    ],
    output: { bands: 1, sampleType: "FLOAT32" }
  }
}
function evaluatePixel(samples) {
  // samples may include .B04, .B08 and optionally .SCL
  var b4 = samples.B04;
  var b8 = samples.B08;
  // safe division
  var ndvi = (b8 + b4) === 0 ? NaN : (b8 - b4) / (b8 + b4);
  var scl = samples.SCL;
  if (scl === undefined || scl === null) {
    return [ndvi];
  }
  // Exclude obvious no-data/cloud classes by returning NaN:
  // SCL classes: 3=cloud_shadow, 9=snow, 10=cloud_high_prob, 11=cloud_medium_prob
  if (scl === 3 || scl === 9 || scl === 10 || scl === 11) {
    return [NaN];
  }
  return [ndvi];
}
"""

EVAL_S2_NDVI_SIMPLE = """\
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04","B08"], units: "REFLECTANCE" }],
    output: { bands: 1, sampleType: "FLOAT32" }
  }
}
function evaluatePixel(s) {
  var b4 = s.B04;
  var b8 = s.B08;
  return [(b8 + b4) === 0 ? NaN : (b8 - b4) / (b8 + b4)];
}
"""

EVAL_QUICK_NDVI = """\
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04","B08"], units: "REFLECTANCE" }],
    output: { bands: 1, sampleType: "UINT8" }
  }
}
function evaluatePixel(s) {
  var red = s.B04;
  var nir = s.B08;
  var ndvi = (nir + red) == 0 ? 0 : (nir - red) / (nir + red);
  // scale from [-1,1] to [0,255]
  var scaled = Math.round(((ndvi + 1.0) / 2.0) * 255.0);
  if (scaled < 0) scaled = 0;
  if (scaled > 255) scaled = 255;
  return [scaled];
}
"""

EVAL_S2_NDVI_COLORMAP = """\
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04","B08"], units: "REFLECTANCE" }],
    // return 3 UINT8 bands ready to be encoded as PNG by the Process API
    output: { bands: 3, sampleType: "UINT8" }
  };
}
function evaluatePixel(s) {
  var b4 = s.B04;
  var b8 = s.B08;
  var ndvi = (b8 + b4) === 0 ? -1.0 : (b8 - b4) / (b8 + b4);

  // Color mapping:
  if (ndvi < 0.0) {
    return [128, 128, 128];
  }
  if (ndvi < 0.2) {
    return [255, 0, 0];
  }
  if (ndvi < 0.4) {
    return [255, 166, 0];
  }
  if (ndvi < 0.6) {
    return [255, 255, 0];
  }
  if (ndvi < 0.8) {
    return [51, 204, 51];
  }
  return [0, 255, 0];
}
"""
