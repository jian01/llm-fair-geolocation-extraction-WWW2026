import sys, json

def load(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            iso2 = cols[0].strip().upper()   # e.g., "AR"
            iso3 = cols[1].strip().upper()   # e.g., "ARG"
            name = cols[4].strip()           # e.g., "Argentina"
            if iso2:
                m[iso2] = {"name": name, "iso3": iso3}
    return m

src = "countryInfo.txt"
cmap = load(src)
# Pretty, stable order; valid JS via JSON literal
js_literal = json.dumps(cmap, ensure_ascii=False, sort_keys=True, indent=2)
print(f"// Auto-generated from {src}")
print("const COUNTRY_MAP = " + js_literal + ";")