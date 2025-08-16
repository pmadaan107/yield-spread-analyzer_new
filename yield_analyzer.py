import requests, numpy as np

# BoC (Canada) – latest selected bond yields via Valet
boc_series = ["V39056","V39059","V39057","V39060","V39058","V39062"]
url_ca = "https://www.bankofcanada.ca/valet/observations/{}/json?recent=1".format(",".join(boc_series))
r_ca = requests.get(url_ca, timeout=10, headers={"User-Agent":"colab-test/1.0"})
print("BoC status:", r_ca.status_code, "bytes:", len(r_ca.content))

# UST (USA) – XML feed, current month
from datetime import date
import xml.etree.ElementTree as ET
ym = date.today().strftime("%Y%m")
url_us = ("https://home.treasury.gov/resource-center/data-chart-center/"
          "interest-rates/pages/xml?data=daily_treasury_yield_curve"
          f"&field_tdr_date_value_month={ym}")
r_us = requests.get(url_us, timeout=10, headers={"User-Agent":"colab-test/1.0"})
print("UST status:", r_us.status_code, "bytes:", len(r_us.content))

# quick parse check (no crash)
ET.fromstring(r_us.content)
print("XML parse OK")
