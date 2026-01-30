from typing import Dict, Any, Optional, List
import io
import zipfile
import re
import pandas as pd

class DataFrameGeoNamesSearcher:

    KEEP_COLUMNS = ["geonameid","name","alternatenames","country_code","feature_code"]
    GEONAMES_COLUMNS = [
        "geonameid","name","asciiname","alternatenames","latitude","longitude",
        "feature_class","feature_code","country_code","cc2","admin1_code",
        "admin2_code","admin3_code","admin4_code","population","elevation",
        "dem","timezone","modification_date"
    ]

    def __init__(self, geonames_path: str):
        self.geonames_df = self.load_geonames_file(geonames_path)

    @staticmethod
    def load_geonames_file(path: str) -> pd.DataFrame:
        if ".zip" in path:
            with open(path, "rb") as f:
                zf = zipfile.ZipFile(io.BytesIO(f.read()))
                df = pd.read_csv(zf.open('allCountries.txt'), sep='\t', header=None, dtype=str, low_memory=False)
                zf.close()
        else:
            with open(path, "rb") as f:
                df = pd.read_csv(path, sep='\t', header=None, dtype=str, low_memory=False)
        df.columns = DataFrameGeoNamesSearcher.GEONAMES_COLUMNS
        for col in ["geonameid"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        #df["name"] = df["name"].str.lower()
        return df

    def get(self, geonameid: int) -> Dict[str, Any]:
        row = self.geonames_df[self.geonames_df["geonameid"] == int(geonameid)]
        if row.empty:
            raise KeyError
        return row[self.KEEP_COLUMNS].iloc[0].to_dict()

    def search(self, query: str, country: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        # 1) Filtro por país (sin upper() por serie; los códigos ya son 'US','BR', etc.)
        d = self.geonames_df
        if country:
            d = d[d["country_code"].fillna("") == country.upper()]
        if d.empty:
            return []

        # 2) Filtros parciales BARATOS (sin regex) para reducir universo
        name_part = d["name"].str.contains(q, case=False, regex=False, na=False)
        asc_part  = d["asciiname"].str.contains(q, case=False, regex=False, na=False) if "asciiname" in d.columns else False
        alt_part  = d["alternatenames"].str.contains(q, case=False, regex=False, na=False)

        sub_mask = name_part | asc_part | alt_part
        if not sub_mask.any():
            return []

        sub = d.loc[sub_mask, ["geonameid","name","asciiname","alternatenames",
                            "country_code","timezone","feature_class","feature_code"]]

        # 3) Exactos:
        #    - name/asciiname exactos case-insensitive (casefold eq)
        q_cf = q.casefold()
        name_exact = sub["name"].astype("string").str.casefold().eq(q_cf)
        asci_exact = sub["asciiname"].astype("string").str.casefold().eq(q_cf) if "asciiname" in sub.columns else False

        #    - token exacto en alternatenames (regex) SOLO sobre los que ya tenían el substring
        alt_candidates = sub["alternatenames"].astype("string")
        alt_token = pd.Series(False, index=sub.index)
        has_alt = alt_candidates.notna() & alt_candidates.str.len().gt(0)
        # restringimos a filas que ya pasaron alt_part para evitar regex sobre todo el subset
        alt_subset_idx = sub.index[alt_part.loc[sub.index] & has_alt]
        if len(alt_subset_idx) > 0:
            pat = rf"(?:^|,)\s*{re.escape(q)}\s*(?:,|$)"
            alt_token.loc[alt_subset_idx] = alt_candidates.loc[alt_subset_idx].str.contains(
                pat, case=False, regex=True, na=False
            )

        exact_mask = name_exact | (asci_exact if isinstance(asci_exact, pd.Series) else False) | alt_token

        # 4) Países primero entre exactos (A.PCL*)
        is_country = (sub["feature_class"] == "A") & sub["feature_code"].astype("string").str.startswith("PCL", na=False)
        exact_sub  = sub.loc[exact_mask]
        country_rows = exact_sub.loc[is_country.reindex(exact_sub.index, fill_value=False),
        DataFrameGeoNamesSearcher.KEEP_COLUMNS].to_dict(orient="records")
        exact_rows   = exact_sub.loc[~is_country.reindex(exact_sub.index, fill_value=False),
        DataFrameGeoNamesSearcher.KEEP_COLUMNS].to_dict(orient="records")

        # 5) Parciales no exactos (del subset)
        partial_rows = sub.loc[~exact_mask, DataFrameGeoNamesSearcher.KEEP_COLUMNS].head(limit * 3).to_dict(orient="records")

        # 6) Merge + dedup + tope
        out: List[Dict[str, Any]] = []
        seen = set()
        for block in (country_rows, exact_rows, partial_rows):
            for r in block:
                try:
                    gid = int(r.get("geonameid"))
                except Exception:
                    continue
                if gid in seen:
                    continue
                out.append(r)
                seen.add(gid)
                if len(out) >= limit:
                    return out
        return out