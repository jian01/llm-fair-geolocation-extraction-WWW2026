from abc import abstractmethod
from typing import Dict, Any, Optional, List



class GeonamesSearcher:
    @abstractmethod
    def search(self, query: str, country: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searchs a text with optional country
        :param limit: the max number of results
        :param query: the text to query
        :param country: the country
        :return: a list of results
        """

    @abstractmethod
    def get(self, geonameid: int) -> Dict[str, Any]:
        """
        Gets a geoname record by id
        :param geonameid: the id
        :return: the record
        """