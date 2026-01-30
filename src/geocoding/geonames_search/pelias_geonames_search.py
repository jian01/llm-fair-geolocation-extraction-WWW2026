import requests
from typing import Dict, Any, Optional, List


LAYERS= "continent,empire,country,dependency,macroregion,region,locality,localadmin,borough,county,macrocounty,neighbourhood,postalcode,street,address,venue,coarse"


class PeliasGeoNamesSearcher:

    KEYS_TO_KEEP = {"id", "layer","name","feature_code"}

    def __init__(self):
        self.known_ids = {}

    def _fetch_from_pelias(self, geonameid: int) -> Optional[Dict[str, Any]]:
        """Internal method to fetch a geonameid from Pelias API."""
        try:
            # Pelias place endpoint accepts ids parameter with geonameid
            req = requests.get(
                "http://localhost:3100/v1/place",
                params={"ids": f"geonameid:{geonameid}"}
            ).json()
            
            if req.get("features") and len(req["features"]) > 0:
                feature = req["features"][0]
                props = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                
                # Extract coordinates if available
                coords = geometry.get("coordinates", [])
                lat, lon = None, None
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # GeoJSON format: [lon, lat]
                
                # Build result dict
                result = {k: v for k, v in props.items() if k in self.KEYS_TO_KEEP}
                
                # Store with lat/lon in known_ids
                stored_result = result.copy()
                if lat is not None and lon is not None:
                    stored_result["latitude"] = lat
                    stored_result["longitude"] = lon
                
                self.known_ids[geonameid] = stored_result
                return result
        except Exception:
            pass
        return None

    def _get(self, geonameid: int) -> Dict[str, Any]:
        """Internal method that returns data WITH latitude and longitude."""
        if geonameid not in self.known_ids:
            self._fetch_from_pelias(geonameid)
        return self.known_ids.get(geonameid, {})

    def get(self, geonameid: int) -> Dict[str, Any]:
        """Get geonameid data without lat/lon. Fetches from Pelias if not in cache."""
        if geonameid not in self.known_ids:
            self._fetch_from_pelias(geonameid)
        
        # Raise KeyError if still not found (maintains backward compatibility)
        if geonameid not in self.known_ids:
            raise KeyError(geonameid)
        
        # Return without lat/lon
        result = self.known_ids[geonameid]
        return {k: v for k, v in result.items() if k not in {"latitude", "longitude"}}

    def search(self, query: str, country: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        keys_to_keep = {"id", "layer", "name", "feature_code", "country_code"}
        if country is None:
            params = {"text": query}
        else:
            params = {"text": query, "boundary.country": country}
        results = []
        features_data = []  # Store full feature data to access geometry
        seen_ids = set()
        for l in LAYERS.split(","):
            req = requests.get("http://localhost:3100/v1/autocomplete", params={"layers": l, **params}).json()
            for r in req["features"]:
                if r["properties"]['id'] in seen_ids:
                    continue
                results.append(r["properties"])
                features_data.append(r)  # Store full feature for geometry access
                seen_ids.add(r["properties"]['id'])
        req = requests.get("http://localhost:3100/v1/autocomplete", params={"text": query, "layers": "country"}).json()
        for r in req["features"]:
            if r["properties"]['id'] in seen_ids:
                continue
            results.append(r["properties"])
            features_data.append(r)  # Store full feature for geometry access
            seen_ids.add(r["properties"]['id'])
        results_w_fc = []
        features_by_id = {f["properties"]["id"]: f for f in features_data}
        for r in results:
            try:
                r['feature_code'] = r['addendum']['geonames']['feature_code']
                results_w_fc.append(r)
            except Exception:
                results_w_fc.append(r)
        results_w_fc = [{k: v for k, v in r.items() if k in keys_to_keep} for r in results_w_fc]
        results_w_fc = sorted(results_w_fc, key=lambda x: -1 if 'feature_code' in x and (
                x['feature_code'] == 'PPLC' or x['feature_code'] == 'PCLI' or 'ADM2' in x['feature_code']) else 1)
        for r in results_w_fc:
            # Store with lat/lon in known_ids
            stored_result = r.copy()
            feature = features_by_id.get(r["id"])
            if feature and "geometry" in feature:
                coords = feature["geometry"].get("coordinates", [])
                if len(coords) >= 2:
                    stored_result["longitude"] = coords[0]  # GeoJSON format: [lon, lat]
                    stored_result["latitude"] = coords[1]
            self.known_ids[int(r["id"])] = stored_result
        return results_w_fc[:limit]