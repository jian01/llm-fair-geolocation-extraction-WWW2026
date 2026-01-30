import re
import json
import ast
import unidecode
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from nltk.tokenize import RegexpTokenizer

try:
    from whoosh.index import open_dir
    from whoosh.qparser import QueryParser
    _WHOOSH_AVAILABLE = True
except Exception:
    _WHOOSH_AVAILABLE = False

tokenizer = RegexpTokenizer(r'\w+')

def safe_parse_alternatenames(x):
    try:
        return ast.literal_eval(x)
    except:
        try:
            return json.loads(x)
        except:
            return []

def word_ngrams(tokens, ngrams):
    exclude = ['province', 'town', 'village', 'hospital', 'sea port', 'port', 'district', \
           'island', 'state', 'municipality', 'governorate', 'axis', 'region', \
           'hub', 'continent', 'idp camp', 'refugee camp', 'sub-district',\
           'subdistrict', 'sub district', 'residential complex', 'basin', 'administrative centre', \
           'administrative center', 'detention centre', 'detention center', 'centre', 'center', \
           'subdivision', 'oil refinery', 'territory', 'lga', 'division', 'country', 'checkpoint', \
           'river', 'market', 'department', 'airport', 'local government area', 'bus station', \
           'military camp', 'highway', 'extension', 'capital', 'border point', 'crossing point', \
           'border', 'road', 'channel', 'mountain', 'prefecture', 'volcano', 'highland', 'lake', \
           'al', 'and', 'sub', 'primary school', 'university city', 'university', 'city', 'school', 'camp', 'area', 'station', \
              'the', 'national', 'park', 'central bank']
    min_n, max_n = 1, ngrams
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                if not " ".join(original_tokens[i: i + n]) in exclude:
                    tokens.append(" ".join(original_tokens[i: i + n]))
    return tokens

def gl_find_names(text, locdictionary = None, locationdata = None, decode = False):
    if decode:
        text = re.sub("-|'", "", unidecode.unidecode(text))
    tokens = word_ngrams(tokenizer.tokenize(text.lower()), 5)
    m = set()
    for token in tokens:
        if token in locdictionary:
            m.add(token)
    # filter out matched places that are substrings of another matched place
    k_list = list(m)
    for i, k in enumerate(k_list):
        for k2 in k_list[:i]:
            if k in k2 and k in m:
                m.remove(k)
        for k2 in k_list[i+1:]:
            if k in k2 and k in m:
                m.remove(k)
    return m


class WhooshSearcher:
    """Reusable Whoosh searcher to avoid repeated index loading overhead."""
    
    def __init__(self, indexdir: str):
        """Initialize with index directory path."""
        self.indexdir = indexdir
        self._ix = None
        self._parser = None
        
    def _ensure_loaded(self):
        """Lazy load the index and parser."""
        if self._ix is None and _WHOOSH_AVAILABLE:
            try:
                self._ix = open_dir(self.indexdir)
                self._parser = QueryParser("place", self._ix.schema)
            except Exception:
                self._ix = None
                self._parser = None
    
    def search(self, query_text: str) -> List[int]:
        """Search for geoname IDs matching the query text."""
        if not _WHOOSH_AVAILABLE or self._ix is None:
            return []
            
        self._ensure_loaded()
        if self._parser is None:
            return []
            
        try:
            with self._ix.searcher() as searcher:
                query = self._parser.parse(query_text)
                results = searcher.search(query)
                return [int(r['geonameid']) for r in results if 'geonameid' in r]
        except Exception:
            return []
    
    def close(self):
        """Close the index (optional cleanup)."""
        if self._ix is not None:
            try:
                self._ix.close()
            except Exception:
                pass
            self._ix = None
            self._parser = None


def load_geonames(locationdata_path: Optional[str] = None,
                  locdictionary_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Load GeoNames tables similarly to existing loaders, but safe for single use.

    Parameters
    ----------
    locationdata_path: Optional[str]
        TSV path with GeoNames entries (expects 'geonameid' index, 'latitude', 'longitude', 'featurecode', 'countrycode', 'population', 'alternatenames').
    locdictionary_path: Optional[str]
        JSON path mapping lowercased place strings to lists of geoname ids.

    Returns
    -------
    (locationdata, locdictionary)
    """
    locationdata = pd.read_csv(locationdata_path, sep='\t', low_memory=False, index_col="geonameid")
    locationdata.loc[locationdata['alternatenames'].isnull(),'alternatenames'] = '[]'
    locationdata["alternatenames"] = locationdata["alternatenames"].apply(safe_parse_alternatenames)

    with open(locdictionary_path, 'r', encoding='utf-8') as f:
        locdictionary = json.load(f)

    return locationdata, locdictionary


def normalize_toponym(name: str) -> str:
    """Normalize a toponym string for robust dictionary matching.

    - Unicode fold
    - Lowercase
    - Remove hyphens and apostrophes
    - Collapse whitespace
    """
    s = unidecode.unidecode(name)
    s = s.lower()
    s = re.sub("-|'", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_lists_noun_modifiers(string: str, plurals: Dict[str, str]) -> List[str]:
    """Split multi-location mentions with noun modifiers.
    
    Example: "New York and Boston cities" -> ["New York city", "Boston city"]
    
    Parameters
    ----------
    string: str
        Input text that may contain multiple locations with a shared noun.
    plurals: Dict[str, str]
        Mapping from plural nouns to singular forms.
        
    Returns
    -------
    List[str]
        List of individual location strings.
    """
    if not string:
        return [string]
        
    plurals_list = list(plurals.keys())
    singulars_list = list(plurals.values())
    output = []
    
    try:
        # Split on common separators
        ls = re.split(r', and |,| and ', string.lower())
        ls = [elem.strip() for elem in ls]
        
        if len(ls) > 1:
            # Try to find plural noun in the last element
            noun = None
            for ele in plurals_list:
                if ' ' + ele in ls[-1].lower():
                    noun = ele
                    break
            
            if noun:
                # Add singular form to each location
                ls = [elem + ' ' + plurals[noun] for elem in ls]
                # Fix the last one by removing the plural and adding singular
                ls[-1] = re.split(' ' + noun, ls[-1], flags=re.IGNORECASE, maxsplit=1)[0] + ' ' + plurals[noun]
            else:
                # Try singular forms
                for ele in singulars_list:
                    if ' ' + ele in ls[-1].lower():
                        noun = ele
                        break
                if noun:
                    ls = [(elem + ' ' + noun).strip() for elem in ls]
                    ls[-1] = (re.split(' ' + noun, ls[-1], flags=re.IGNORECASE, maxsplit=1)[0] + ' ' + noun).strip()
        
        output.extend(ls)
    except Exception:
        output.append(string.strip())
    
    return output


def split_lists_possessive_pronouns(string: str, plurals: Dict[str, str]) -> List[str]:
    """Split multi-location mentions with possessive pronouns.
    
    Example: "Cities of New York and Boston" -> ["New York city", "Boston city"]
    
    Parameters
    ----------
    string: str
        Input text with possessive pattern.
    plurals: Dict[str, str]
        Mapping from plural nouns to singular forms.
        
    Returns
    -------
    List[str]
        List of individual location strings.
    """
    if not string:
        return [string]
        
    output = []
    
    try:
        if '(' not in string:
            # Handle special case for "Trinidad and Tobago"
            entry_ = string.lower().replace("trinidad and tobago", "trinidadandtobago")
            
            # Split on " of " to get noun and locations
            parts = re.split(' of ', entry_, 1)
            if len(parts) == 2:
                noun = parts[0].strip()
                locations_part = parts[1].strip()
                
                # Split locations on separators
                locations = re.split(r', and |,| and ', locations_part)
                if len(locations) > 1:
                    # Add noun to each location
                    if noun.lower() in plurals:
                        singular_noun = plurals[noun.lower()]
                    else:
                        singular_noun = noun.lower()
                    
                    for loc in locations:
                        clean_loc = loc.replace("trinidadandtobago", "trinidad and tobago")
                        output.append(clean_loc + ' ' + singular_noun)
                else:
                    output.append(string)
            else:
                output.append(string)
        else:
            output.append(string)
    except Exception:
        output.append(string)
    
    return output


def split_lists_cardinals(string: str, cardinals: List[str]) -> List[str]:
    """Split multi-location mentions with cardinal directions.
    
    Example: "north, south, east regions" -> ["north region", "south region", "east region"]
    
    Parameters
    ----------
    string: str
        Input text with cardinal directions.
    cardinals: List[str]
        List of cardinal direction terms.
        
    Returns
    -------
    List[str]
        List of individual location strings.
    """
    if not string:
        return [string]
        
    output = []
    
    try:
        # Split on separators
        ls = re.split(r', and |,| and ', string.lower())
        ls = [elem.strip() for elem in ls]
        
        if len(ls) > 1:
            # Check if all but last are cardinals
            first_part_cardinals = all(elem.strip().lower() in cardinals for elem in ls[:-1])
            last_part_cardinals = all(elem.strip().lower() in cardinals for elem in ls[1:])
            
            if first_part_cardinals and len([elem.strip().lower() in cardinals for elem in ls[:-1]]) > 0:
                # Find noun in last element
                right_cardinal = None
                for card in cardinals:
                    if card in ls[-1].lower():
                        right_cardinal = card
                        break
                
                if right_cardinal:
                    noun = re.split(right_cardinal, ls[-1], flags=re.IGNORECASE, maxsplit=1)[1].strip()
                    output = [elem.strip() + ' ' + noun for elem in ls[:-1]]
                    output.append(right_cardinal + ' ' + noun)
                else:
                    output.append(string)
            elif last_part_cardinals and len([elem.strip().lower() in cardinals for elem in ls[1:]]) > 0:
                # Find noun in first element
                left_cardinal = None
                for card in cardinals:
                    if card in ls[0].lower():
                        left_cardinal = card
                        break
                
                if left_cardinal:
                    noun = re.split(left_cardinal, ls[0], flags=re.IGNORECASE, maxsplit=1)[0].strip()
                    output.append(left_cardinal + ' ' + noun)
                    output.extend([elem.strip() + ' ' + noun for elem in ls[1:]])
                else:
                    output.append(string)
            else:
                output.append(string)
        else:
            output.append(string)
    except Exception:
        output.append(string)
    
    return output


def split_multi_location_mention(toponym: str) -> List[str]:
    """Split a toponym that may contain multiple locations into individual locations.
    
    This function applies the three splitting strategies from the original codebase:
    1. Noun modifiers: "New York and Boston cities"
    2. Possessive pronouns: "Cities of New York and Boston"  
    3. Cardinal directions: "north, south, east regions"
    
    Parameters
    ----------
    toponym: str
        Input text that may contain multiple locations.
        
    Returns
    -------
    List[str]
        List of individual location strings.
    """
    # Define plurals mapping (from original code)
    plurals = {
        "provinces": "province", "towns": "town", "villages": "village", "hospitals": "hospital",
        "sea ports": "sea port", "ports": "port", "districts": "district", "cities": "city",
        "islands": "island", "states": "state", "municipalities": "municipality",
        "governorates": "governorate", "axes": "axis", "regions": "region", "hubs": "hub",
        "continents": "continent", "idp camps": "idp camp", "refugee camps": "refugee camp",
        "camps": "camp", "areas": "area", "sub-districts": "sub-district",
        "subdistricts": "subdistrict", "sub districts": "sub district",
        "residential complexes": "residential complex", "basins": "basin",
        "administrative centres": "administrative centre", "administrative centers": "administrative center",
        "detention centres": "detention centre", "detention centers": "detention center",
        "centres": "centre", "centers": "center", "subdivisions": "subdivision",
        "oil refineries": "oil refinery", "territories": "territory", "lgas": "lga",
        "divisions": "division", "countries": "country", "checkpoints": "checkpoint",
        "rivers": "river", "markets": "market", "departments": "department",
        "airports": "airport", "local government areas": "local government area",
        "bus stations": "bus station", "military camps": "military camp",
        "highways": "highway", "extensions": "extension", "capitals": "capital",
        "border points": "border point", "crossing points": "crossing point",
        "borders": "border", "roads": "road", "channels": "channel",
        "mountains": "mountain", "prefectures": "prefecture", "volcanoes": "volcano",
        "volcanos": "volcano", "highlands": "highland", "lakes": "lake"
    }
    
    # Define cardinal directions (from original code)
    cardinals = [
        "northeast", "north-east", "northwest", "north-west", "northouest", "north-ouest",
        "north east", "north west", "south east", "south west", "southeast", "south-east",
        "southwest", "south-west", "southouest", "south-ouest", "north", "east", "west",
        "ouest", "south", "central"
    ]
    
    # Apply splitting strategies in sequence
    result = [toponym]
    
    # 1. Noun modifiers
    result = [item for sublist in [split_lists_noun_modifiers(s, plurals) for s in result] for item in sublist]
    
    # 2. Possessive pronouns  
    result = [item for sublist in [split_lists_possessive_pronouns(s, plurals) for s in result] for item in sublist]
    
    # 3. Cardinal directions
    result = [item for sublist in [split_lists_cardinals(s, cardinals) for s in result] for item in sublist]
    
    # Clean up and remove duplicates while preserving order
    seen = set()
    unique_result = []
    for item in result:
        clean_item = item.strip()
        if clean_item and clean_item not in seen:
            seen.add(clean_item)
            unique_result.append(clean_item)
    
    return unique_result


def match_candidates(toponym: str,
                     locationdata: pd.DataFrame,
                     locdictionary: Dict[str, List[int]],
                     searcher: Optional[WhooshSearcher] = None,
                     filter_search_engine: str = '') -> List[int]:
    """Get candidate geoname IDs for a single toponym using the dictionary and optional Whoosh search.

    Parameters
    ----------
    toponym: str
        Raw input toponym text (single mention or name).
    locationdata: pd.DataFrame
        GeoNames table.
    locdictionary: Dict[str, List[int]]
        String-to-ids dictionary for direct matches.
    searcher: Optional[WhooshSearcher]
        Pre-initialized Whoosh searcher object (reuse across calls).
    filter_search_engine: str
        Regex that 'featurecode' must match to keep search hits (empty means keep all).

    Returns
    -------
    List[int]
        Unique list of candidate geoname IDs.
    """
    norm = normalize_toponym(toponym)

    # 1) Dictionary matches via existing n-gram finder
    dict_names = gl_find_names(norm, locdictionary, locationdata, decode=False)
    dict_ids = set()
    for token in dict_names:
        ids = locdictionary.get(token, [])
        for gid in ids:
            dict_ids.add(int(gid))

    search_ids = set()
    if searcher is not None:
        search_ids = set(searcher.search(norm))
        
        if len(search_ids) > 0 and filter_search_engine:
            hits_df = locationdata[locationdata.index.isin(search_ids)]
            filt = hits_df[hits_df.featurecode.str.contains(filter_search_engine, na=False)]
            search_ids = set(map(int, filt.index.tolist()))

    return list(sorted(dict_ids.union(search_ids)))


def build_candidate_frame(geoname_ids: List[int], locationdata: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of candidate places enriched with core attributes.

    Includes: latitude/longitude (radians), population, countrycode, featurecode.
    """
    if len(geoname_ids) == 0:
        return pd.DataFrame(columns=["geonameid", "latitude", "longitude", "population", "countrycode", "featurecode"])

    df = locationdata[locationdata.index.isin(geoname_ids)].copy()
    df = df[["name", "latitude", "longitude", "population", "countrycode", "featurecode"]]
    df = df.reset_index().rename(columns={"index": "geonameid"})

    # Convert to radians for distance consistency with existing codebase
    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
    df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')

    import numpy as np
    df["latitude"] = np.radians(df["latitude"])
    df["longitude"] = np.radians(df["longitude"])

    # Fill missing population with 0 to avoid NaNs in ranking
    df["population"] = pd.to_numeric(df["population"], errors='coerce').fillna(0)

    return df


def compute_rule_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Compute rule-based features (single toponym, no doc context).

    Features mirror the multi-mention logic but without country distribution:
    - is_country (PCLI/PCLS)
    - is_capital (PPLC)
    - is_city (feature code contains PPL)
    - adm_level extracted from featurecode suffix (default 6, lower is more administrative)
    - population (higher is better)
    """
    if len(candidates) == 0:
        return candidates

    df = candidates.copy()

    df['is_country'] = df.featurecode.isin(["PCLI", "PCLS"]).astype(int)
    df['is_capital'] = df.featurecode.isin(["PPLC"]).astype(int)
    df['is_city'] = df.featurecode.str.contains("PPL", na=False).astype(int)

    # Admin level: extract digits from featurecode like ADM1, ADM2; default 6. PPLA as 1
    df['adm_level'] = df["featurecode"].str.findall(r'\d+').str.join("").replace("", 6)
    df['adm_level'] = df['adm_level'].fillna(6)
    df['adm_level'] = df['adm_level'].astype(int)
    df.loc[df.featurecode == "PPLA", "adm_level"] = 1
    df.loc[df['is_capital'] == 1, 'adm_level'] = 0
    df.loc[df['is_country'] == 1, 'adm_level'] = 0

    return df


def rank_single_toponym(df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates for a single toponym using a rule-based ordering.

    Order by:
    - is_capital desc
    - is_country desc
    - is_city desc
    - lower adm_level first
    - population desc
    Returns a DataFrame with an added 'score_rank' where 1 is best.
    """
    if len(df) == 0:
        return df

    df_ = df.copy()
    df_.sort_values(by=["is_capital", "is_country", "is_city", "adm_level", "population"],
                    ascending=[False, False, False, True, False], inplace=True)
    df_["score_rank"] = range(1, len(df_) + 1)
    return df_


def geolocate_toponym(toponym: str,
                      locationdata: pd.DataFrame,
                      locdictionary: Dict[str, List[int]],
                      searcher: Optional[WhooshSearcher] = None,
                      filter_search_engine: str = '') -> Dict[str, Any]:
    """Geolocate a single toponym using rule-based disambiguation.

    Parameters
    ----------
    toponym: str
        Input place name (may contain multiple locations).
    locationdata, locdictionary: preloaded resources.
    searcher: Optional[WhooshSearcher]
        Pre-initialized Whoosh searcher for search augmentation.
    filter_search_engine: str
        Regex that 'featurecode' must match to keep search hits.
    split_multi_locations: bool
        If True, split multi-location mentions (e.g., "New York and Boston cities").

    Returns
    -------
    dict
        {
          'toponym': str,
          'split_locations': List[str],  # individual locations after splitting
          'best': Optional[dict],  # best candidate with attributes
          'candidates': List[dict] # ranked candidates (possibly empty)
        }
    """
    # Split multi-location mentions if requested
    split_locations = split_multi_location_mention(toponym)
    results = []
    for location in split_locations:
        geoname_ids = match_candidates(location, locationdata, locdictionary, searcher, filter_search_engine)
        if len(geoname_ids) == 0:
            continue
        cand = build_candidate_frame(geoname_ids, locationdata)
        cand = compute_rule_features(cand)
        ranked = rank_single_toponym(cand)

        best_row = ranked.iloc[0].to_dict() if len(ranked) else None
        results.append({
            'toponym': toponym,
            'split_locations': split_locations,
            'best': best_row,
            'candidates': ranked.to_dict(orient='records')
        })
    return results

