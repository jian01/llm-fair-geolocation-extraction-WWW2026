
"""
geonames_agent.py — Single-file GeoNames geolocator (LangChain Tools)
---------------------------------------------------------------------
- Carga GeoNames allCountries.zip
- Construye un agente con LangChain (create_openai_tools_agent)
- CLI para single y batch

Requisitos:
    pip install -U langchain langchain-openai pydantic pandas openai

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set

from psutil.tests import retry
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.geocoding.geonames_search import GeonamesSearcher


# --- Config -----------------------------------------------------------

@dataclass
class Settings:
    model: str = "gpt-4o"
    temperature: float = 0.1
    top_p: float = 1.0
    verbose: bool = False
    return_intermediate_steps: bool = True
    max_iterations: int = 15

PROMPT = """
## Task & Tools

You work as a **geolocator**. A location (PLACE) and its **context** will be provided. Your goal is to maximize 100 mile accuracy.
Use the **tools**
1) `search_tool` candidates in GeoNames database. You can pass an ISO country code when the context hints at it (e.g., 'AR', 'BR', 'MZ'),
2) For each **definite** location, call **`select_tool`** (you may call it multiple times) for select the best matching entry. Be precise and prefer cities/settlements over vague regions when the context points to them, 
3) When you are done, call **`finish_tool`** exactly once to return **all selections together**.

## Considerations

- If PLACE is ambiguous, refine with `search_tool` (try alternate spellings from context).
- When PLACE is a country name, you MUST select the sovereign state entry (feature_code PCL*).
- If a PCL* candidate is not returned on the first search, try alternate spellings or exonyms/endonyms and search again.
- If the context indicates a country explicitly, include it in searches.
- Prefer inhabited places (feature classes `P*`) when context refers to education, health, local news, municipal services, etc.
- If the text clearly references a **province/state/county** instead of a city, select that administrative division.
- Avoid false positives from homonyms in other countries—check language, nearby mentions, and timezone cues in context.
- If no good candidate appears, perform another `search_tool` with a better query (e.g., remove accents, try shorter stems).
- If PLACE matches a sovereign state (country name), prefer the country entry (feature_code = 'PCLI') rather than a city.
- Do not call `select_tool` twice for the same geonameid or the same place.
- You must cover every sublocation enumerated in PLACE; do not call finish_tool until each is either selected or explicitly marked with select_tool(geonameid=-1, ...).
- You only have 15 actions to use.
- Whether relevant or not, try to locate all toponyms, only the ones not in geonames should be left without selection.
- Prefer shorter searches and do not insist too much on one location, you have limited action, prioritize selection before searching.
- Do at most **only two** searches per place.
- Do not extract implicit location from within the context or not explicitly named or directly associated to the place string
- If the toponym is "<X> in <Y>" extract just <X> unless <X> is not in geonames, then extract <Y>
- Do **not** add extra locations not related to the place input

## Non-literal (associative) toponyms

Examples of non-literal toponyms:
* Metonymy: She used to play for **Cambridge**.
* Homonym: I asked **Paris** to help me.
* Demonym: I spoke to a **Jamaican** on the bus.
* Language: She spoked **Spanish**.
* Noum modifier: That **Paris** souvenir is interesting.
* Adjectival modifier: I ate some **Spanish** ham yesterday.
* Embedded associative: **US** Supreme Court has 9 justices.
You should mark those kind of toponyms as not literal.

## Output protocol
- Call `select_tool(geonameid, context, literal_toponym)` for every confirmed location.
  - geonameid should **always** be present as a valid geonames id, get the closest one possible.
  - `context` is an English explanation like "Neighborhood mentioned in Yola North LGA", or
    "Government jurisdiction", etc. If the toponym is not literal you should specify why.
  - `literal_toponym` is True when the toponym refers to a real place related to what the document is describing, False when associative.
  - You should select only one id per place and then finish.
- Finally, call `finish_tool(reason=...)` once. Do not return free-form text.

## Examples
User says:
Place: Cucha Cucha
Context: Este domingo 23 de febrero, desde las 09:00 horas, se llevará a cabo en la localidad de **Cucha Cucha** el festival “Voz de mi pueblo”. En el mismo, habrá concurso de asadores, fogón y desfile criollo, prueba de riendas y sortijas, Autos del Ayer y los talleres abiertos y gratuitos de la Escuela de Actividades Culturales."
You should search for "cucha cucha".
Then, after getting the right geonameid, call the selection tool with that id and a context like "This is where the festival takes place." and it is a literal toponym.

User says:
Place: Tokyo
Context: The **Tokyo** Metropolitan Government announced new measures to reduce traffic congestion ahead of the upcoming holiday season.
You should search for "tokyo".
Then, after getting the right geonameid, call the selection tool with that id and a context like "Is the jurisdiction of a government institution" and it is not a literal toponym.

User says:
Place: Canada
Context: After months of negotiations with international partners, the European Union welcomed the announcement that **Canada** would contribute additional funding to support climate adaptation projects in vulnerable countries.
You should search for "canada".
Then, after getting the right geonameid, call the selection tool with that id and a context like "Canada is the acting government" and it is not a literal toponym.

User says:
Place: Paris
Context: That **Paris** souvenir is interesting.
You should search for "paris".
Then, after getting the right geonameid, call the selection tool with that id and a context like "Is where the souvenir is from." and it is not a literal toponym since it is a noun modifier.

User says:
Place: Doubeli, Luggere and Jambutu of Yola North LGA
Context: "Incidents were reported in ..."
Thought: Search and select each of the three localities (multiple select_tool calls). Then finish_tool.

User says:
Place: Silicon Valley
Context: "In 2023, the tech industry in **Silicon Valley** saw a surge in AI startups"
Thought: search 'silicon valley', if no results search Mountain View since silicon valley probably isn't in geonames

User says:
Place: City of Buenos Aires, Argentina
Context: "In **City of Buenos Aires, Argentina** a big crisis happened in 2001."
Thought: search 'buenos aires', and select the city of buenos aires and finish

User says:
Place: Palermo Hollywood in Buenos Aires
Context: "In the neighborhood of **Palermo Hollywood in Buenos Aires** there are great cafes."
Thought: search 'Palermo Hollywood', since its not in geonames, try "palermo", if its also not in geonames search the city of buenos aires

User says:
Place: Northeast Brazil
Context: "**Northeast Brazil** faces recurrent droughts that strongly shape local agricultural practices."
Thought: search 'Northeast Brazil', since its not in geonames, use "Brazil".
"""


# --- Schemas ----------------------------------------------------------

class SearchArgs(BaseModel):
    query: str = Field(..., description="lowercase query to match name or alternatenames (substring) in geonames database. "
                                        "the query will only return a result if its contained fully in the result."
                                        "**Adding more words to a previous query wont give you more results**")
    country: Optional[str] = Field(None, description="optional ISO alpha-2 country code, e.g., AR, BR, MZ")


class SelectArgs(BaseModel):
    geonameid: int = Field(..., description="target GeoName ID to select. -1 if not found any")
    place: str = Field(..., description="the place for which the geoname id corresponds to")
    context: str = Field(..., description="the context explaining the location, in english")
    literal_toponym: bool = Field(..., description="if the location is a real literal toponym and not an associative one")

class FinishArgs(BaseModel):
    reason: Optional[str] = Field(None, description="optional short reason/summary before finishing")


# --- Agent class ------------------------------------------------------

class GeoNamesAgent:

    def __init__(self, geonames_searcher: GeonamesSearcher, settings: Settings = Settings()) -> None:
        self.geonames_searcher = geonames_searcher
        self.settings = settings
        self._selections: List[Dict[str, Any]] = [] # memoria interna para multi-select

        # tools        
        self.search_tool = StructuredTool.from_function(
            name="search_tool",
            func=lambda query, country=None: self._search_tool(self.geonames_searcher, query, country),
            args_schema=SearchArgs,
            return_direct=False,
            description=self._search_tool.__doc__
        )
        self.select_tool = StructuredTool.from_function(
            name="select_tool",
            func=lambda geonameid, place, context, literal_toponym: self._select_tool(self.geonames_searcher, [geonameid], place,
                                                                                       context, literal_toponym),
            args_schema=SelectArgs,
            return_direct=False,
            description=self._select_tool.__doc__
        )
        self.finish_tool = StructuredTool.from_function(
            name="finish_tool",
            func=lambda reason=None: self._finish_tool(reason),
            args_schema=FinishArgs,
            return_direct=True,
            description=self._finish_tool.__doc__
        )

        self.executor = self._build_executor()

    @staticmethod
    def _search_tool(
        geonames_searcher: GeonamesSearcher,
        query: str,
        country: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Simple substring search en GeoNames por name/asciiname/alternatenames (case-insensitive).
        Search the closest location expected to be in geonames.
        Case insensitive, the query will only return a result if its contained fully in the result.
        - Optional country filter (ISO alpha-2). If no results are given you should try without this.
        """
        return geonames_searcher.search(query, country, limit)


    def _select_tool(self, geonames_searcher: GeonamesSearcher, geonameids: List[int],
                     place: str, context: str, literal_toponym: bool) -> Dict[str, Any]:
        """
        Selects a geoname id to append it to the result
        :param geonameids: the geonames ids, is the closest geoname id available
        :param place: the place name for which the selection is
        :param context: the context that explains the relation between the location and the text
        :param literal_toponym: if the toponym is literal or not
        """
        selected = []

        for gid in geonameids:
            if gid == -1:
                entry = {"context": context, "place": place, "literal_toponym": literal_toponym, "not_found": True}
                selected.append(entry)
                continue

            try:
                r = geonames_searcher.get(gid)
            except KeyError:
                entry = {"error": f"Invalid geonameid: {gid}", "context": context, "place": place, "literal_toponym": literal_toponym}
                selected.append(entry)
                continue
            r["place"] = place
            r["context"] = context
            r["literal_toponym"] = literal_toponym
            selected.append(r)

        self._selections += selected
        return {"selected": selected, "count": len(self._selections)}

    def _finish_tool(self, reason: Optional[str] = None) -> Dict[str, Any]:
        """Returns all accumulated selections and finishes."""
        out = {"selections": self._selections, "finished_reason": reason}
        # reset memory for next run
        self._selections = []
        return out

    def _build_executor(self) -> AgentExecutor:
        llm = ChatOpenAI(model=self.settings.model, temperature=self.settings.temperature, top_p=self.settings.top_p)
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT),
            ("human", "Place: {place}\nContext: {context}"),
            MessagesPlaceholder("agent_scratchpad", optional=True),
        ])
        tools = [self.search_tool, self.select_tool, self.finish_tool]
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.settings.verbose,
            return_intermediate_steps=self.settings.return_intermediate_steps,
            handle_parsing_errors="Fix your format.",
            max_iterations=self.settings.max_iterations
        )
        return executor

    @retry(3)
    def run_agent(self, place: str, context: str) -> Dict[str, Any]:
        self._selections = []
        result = self.executor.invoke({"place": place, "context": context})
        return result.get("output", result)
