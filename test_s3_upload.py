# # test_s3_upload.py
# import numpy as np
# from utils import ndvi_to_png_bytes, make_s3_key, upload_bytes_to_s3, generate_presigned_url

# # make a simple NDVI-like array (values between -1..1)
# arr = np.linspace(-0.5, 0.5, num=10000).reshape((100,100)).astype("float32")
# png = ndvi_to_png_bytes(arr)  # uses your utils function

# key = make_s3_key("test_ndvi", "sample.png")
# print("Uploading to key:", key)
# uploaded_key = upload_bytes_to_s3(png, key, content_type="image/png")
# print("Uploaded key:", uploaded_key)
# url = generate_presigned_url(uploaded_key, expires_in=600)
# print("Presigned URL (valid 10m):", url)
import auth


token = auth.get_sh_token()  # Ensure token fetching works at startup
print("server token prefix:", token)



# import logging

# import requests, json
# from auth import get_sh_token

# LOG = logging.getLogger(__name__)
# token = get_sh_token()
# url = "https://services.sentinel-hub.com/api/v1/process"
# hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
# payload = {
#   "input": {
#     "bounds": {"bbox": [73.784881085, 18.462754539, 73.867333196, 18.483471932]},
#     "data": [{ "type": "sentinel-2-l2a", "dataFilter": {"timeRange":{"from":"2025-11-10T05:25:34Z","to":"2025-11-10T05:25:34Z"}} }]
#   },
#   "evalscript": "//VERSION=3\nfunction setup(){return{input:[{bands:[\"B04\",\"B03\",\"B02\"],units:\"REFLECTANCE\"}],output:{bands:3}}}\nfunction evaluatePixel(s){return [s.B04,s.B03,s.B02];}",
#   "output": {"width":128,"height":128,"responses":[{"identifier":"default","format":{"type":"image/png"}}]}
# }

# r = requests.post(url, headers=hdr, json=payload, allow_redirects=False, timeout=60)
# if r.status_code == 200:
#     ctype = r.headers.get("Content-Type", "")
#     if ctype.startswith("application/json"):
#         txt = (r.text or "").strip()
#         if txt == "null" or txt == "[]":
#             LOG.warning("Process API returned JSON null/empty for payload; trying fallback or catalog-based processing")
#             # try bbox fallback (you already do), or try evalscript simpler
#             # or fall back to catalog assets method (see below)

# print("status:", r.status_code)
# print("url:", r.url)
# print("resp headers:", r.headers)
# print("resp body:", (r.text or r.content)[:1000])
