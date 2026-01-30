from src.llm_json_tagging import _result_parse, _split_text_evenly, coerce_tags
from .dataset import Tag, Document
from typing import List, Optional
import json
import re
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

GPT_AGENT_PROMPT = """
You are a Named Entity Recognition (NER) system specialized in extracting **literal toponyms** (geographic location names) from texts about natural disasters and accidents. Your task is to identify and return **only the explicit literal mentions of physical locations** (toponyms), avoiding any associative uses.
Each time the user gives you a text you simply answer each occurence of an explicit literal toponym avoiding associative toponyms.
You will sometimes receive feedback from missing or incorrect entities from other agents, **give this fixes a high priority**.

Key constraints:
- Extract only literal toponyms, defined as:
  - Proper names of places (e.g., "Cambridge", "Germany")
  - Noun modifiers of places (e.g., "Paris pub")
  - Adjectival modifiers with geographic meaning (e.g., "southern Spanish city")

- Do NOT extract associative toponyms, including:
  - Metonymic references (e.g., "She used to play for Cambridge")
  - Demonyms (e.g., "a Jamaican")
  - Homonyms (e.g., "I asked Paris to help")
  - Languages (e.g., "in Spanish")
  - Noun/adjectival modifiers not referring to a physical place (e.g., "Spanish ham")
  - Embedded uses (e.g., "US Supreme Court", "US Dollar")
  - Toponyms in URLs

Extraction Rules:
1. Preserve literal mentions exactly as they appear — no rephrasing or normalization.
2. Preserve order — output the locations in the same order as in the input.
3. Do not remove duplicates.
4. Include:
   - Geographical regions (e.g., "Patagonia", "coastal Germany")
   - Roads, borders, and composite names (e.g., "Buenos Aires–Mar del Plata road")
   - Temporary places like refugee camps
   - Institutions only if they imply a geographic location (e.g., "Cambridge University" implies "Cambridge")
5. Prefer the most specific geographic level available (e.g., "Buenos Aires province" over "Buenos Aires").
6. Include articles if they are part of the place name.
7. Include **cardinal directions** (e.g., "southern Spain").
8. **Do not merge** multiple toponyms unless they jointly modify a noun immediately after (e.g., “South Dakota, New York and Michigan states” → merge all three).
9. If multiple locations are listed and are connected by commas, “and”, or “in” within a single continuous phrase, keep them together as one literal toponym exactly as written, even if they contain multiple place names.
10. Merge nested location phrases with possessive or relational structure
11. Ignore markdown styles

Input format:
You receive a raw paragraph of text. You may also receive feedback from another verifier agent to fix some errors.

Output format:
Return a JSON list of strings, each being a literal toponym extracted from the text. The output should be a valid JSON with no extra chars.

Examples:

Input: 
Verifier feedback: Merge "Germany and Brazil" as one tag.
"In 2023, the tech industry in **Silicon Valley** saw a surge in AI startups, with major investments coming from firms based in **New York and London**.

Meanwhile, renewable energy projects in **Germany** and **Brazil** gained momentum, with new wind farms being developed in coastal regions.

**Tokyo** hosted a major international summit on climate change, while **Shanghai** continued to expand its smart city infrastructure."
Output:
["Silicon Valley", "New York and London", "Germany and Brazil", "Tokyo", "Shanghai"]
Input: "The descendants of Welsh immigrants who set sail to Argentina 160 years ago have planted 1,500 daffodils as a nod to their roots.

About 150 immigrants travelled to Patagonia on a converted tea clipper ship from Liverpool to Puerto Madryn in 1865, a journey that took two months."

Output:
["Argentina", "Patagonia", "Liverpool", "Puerto Madryn"]
Input: 
"Evacuation orders were issued in Northern Nevada, Western Idaho and Central Montana counties following a series of wildfires.
A team from the University of Oregon arrived on site, supported by volunteers from Canada and Australia.
Among them was a Canadian who had previously worked on wildfires in Asia.
Heatwaves in sub-Saharan Africa have also intensified migration toward Mediterranean shores."
Output:
["Northern Nevada, Western Idaho and Central Montana counties", "University of Oregon", "Canada and Australia", "Asia", "sub-Saharan Africa", "Mediterranean shores"]
Input:
"A fire broke out near Jungle Base Alpha, located just outside Kinshasa.
Reports came in from travelers between Lusaka and Harare, describing heavy smoke along the Great North Road.
New Zealand’s ambassador said Māori communities have been particularly affected.
Meanwhile, the Tokyo-based firm Osaka Robotics donated masks and tents.
Videos posted on www.relief-africa.org showed camps in Western Sahara being evacuated."
Output: 
["Jungle Base Alpha", "Kinshasa", "Lusaka and Harare", "Great North Road", "camps in Western Sahara"]
Input:
"Flash floods swept through Albay, Camarines Sur and Sorsogon provinces overnight, leaving several villages inaccessible.
In the Chitral district of Khyber Pakhtunkhwa province, emergency shelters were set up by local authorities."
Output: 
["Albay, Camarines Sur and Sorsogon provinces", "Chitral district of Khyber Pakhtunkhwa province"]
Input:
"Flash flooding affected the Hlaingthaya district, Yangon Region, displacing hundreds overnight.
In the Deir ez-Zor district of the Deir ez-Zor Governorate, emergency services struggled to reach remote communities due to damaged infrastructure."
Output:
["Hlaingthaya district, Yangon Region", "Deir ez-Zor district of the Deir ez-Zor Governorate"]
Input:
"Rescue teams worked in the coastal town in southern Portugal to assist residents after the earthquake."
Output:
["coastal town in southern Portugal"]
Input: 
Verifier feedback: 'Arras district, Beaurains, Dainville and Saint-Nicolas areas in Pas-de-Calais district' should be one unique tag
"Fighting intensified overnight in Arras district, Beaurains, Dainville and Saint-Nicolas areas in Pas-de-Calais district, with local authorities reporting multiple incidents of damage to infrastructure. Emergency services were deployed to assist residents, while security forces established checkpoints along the main access roads. Officials have urged civilians to remain indoors until further notice as operations continue."
Output:
["Arras district, Beaurains, Dainville and Saint-Nicolas areas in Pas-de-Calais district"]
"""

VERIFIER_PROMPT = """
You are a verifier for a NER system.
Task: Decide whether the tagging done by the NER system captures all the entities without missing any one and has done all the mergers of entities. 
Do **NOT EVER suggest splitting entities**. Do not suggest removing repetitions.

Key constraints:
- Extract only literal toponyms, defined as:
  - Proper names of places (e.g., "Cambridge", "Germany")
  - Noun modifiers of places (e.g., "Paris pub")
  - Adjectival modifiers with geographic meaning (e.g., "southern Spanish city")


- Do NOT extract associative toponyms, including:
  - Metonymic references (e.g., "She used to play for Cambridge")
  - Demonyms (e.g., "a Jamaican")
  - Homonyms (e.g., "I asked Paris to help")
  - Languages (e.g., "in Spanish")
  - Noun/adjectival modifiers not referring to a physical place (e.g., "Spanish ham")
  - Embedded uses (e.g., "US Supreme Court", "US Dollar")
  - Toponyms in URLs
  
Extraction Rules:
1. Include:
   - Geographical regions (e.g., "Patagonia", "coastal Germany")
   - Roads, borders, and composite names (e.g., "Buenos Aires–Mar del Plata road")
   - Temporary places like refugee camps
   - Institutions only if they imply a geographic location (e.g., "Cambridge University" implies "Cambridge")
2. Prefer the most specific geographic level available (e.g., "Buenos Aires province" over "Buenos Aires").
3. Include articles if they are part of the place name.
4. Include **cardinal directions** (e.g., "southern Spain").
5. **Do not merge** multiple toponyms **unless** they jointly modify a noun immediately after (e.g., “South Dakota, New York and Michigan states” → merge all three).
6. If multiple locations are listed and are connected by commas, “and”, or “in” within a single continuous phrase, keep them together as one literal toponym exactly as written, even if they contain multiple place names.
7. Merge nested location phrases with possessive or relational structure
8. **No entity should be split** into multiple entities by the verifier.
9. The suggested entities to extract should be literal strings contained in the text.

You will receive:
- HIGHLIGHTED_TEXT: the original input text with **bold** spans for the entities

Output format (strict):
{{"is_correct": true|false, "reason": "<brief justification>"}}

Data:
HIGHLIGHTED_TEXT (visual aid):
<<<{HIGHLIGHTED_TEXT}>>>

Examples:
Input:
HIGHLIGHTED_TEXT (visual aid):
<<<Reports came in from travelers between **Lusaka** and Harare, describing heavy smoke along the **Great North Road**.>>>
Output:
{{"is_correct": false, "reason": "Harare is missing from the tags"}}
Input:
HIGHLIGHTED_TEXT (visual aid):
<<<Severe winter storms impacted **South Dakota**, **New York** and **Michigan** states, causing widespread power outages and travel disruptions.>>>
Output:
{{"is_correct": false, "reason": "'South Dakota, New York and Michigan states' should be one unique tag"}}
Input:
HIGHLIGHTED_TEXT (visual aid):
<<<Many farmers in the **south** faced prolonged drought conditions this year, affecting crop yields and local economies.>>>
Output:
{{"is_correct": false, "reason": "'south' should not be a tag since its not an explicit location"}}
Input:
HIGHLIGHTED_TEXT (visual aid):
<<<Authorities temporarily closed the road from Buenos Aires to Mar del Plata after heavy flooding.>>>
Output:
{{"is_correct": false, "reason": "'the road from Buenos Aires to Mar del Plata' should be one single location"}}
HIGHLIGHTED_TEXT (visual aid):
<<<Aid groups expanded shelters at the **Al-Hol** refugee camp in northeastern **Syria** after flash floods damaged hundreds of tents.>>>
Output:
{{"is_correct": false, "reason": "Both 'Al-Hol refugee camp' and 'northeastern Syria' should be captured as more specific tags"}}
"""

VERIFIER_DATA = """
HIGHLIGHTED_TEXT (visual aid):
<<<{HIGHLIGHTED_TEXT}>>>
"""

extractor_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
verifier_llm  = ChatOpenAI(model="gpt-4o", temperature=0.1)

extractor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         GPT_AGENT_PROMPT),
        MessagesPlaceholder("history"),
        ("human",
         "Verifier feedback: {feedback}\n"
         "\"{RAW_TEXT}\"")
    ]
)
extractor_chain = extractor_prompt | extractor_llm

class Verdict(BaseModel):
    is_correct: bool = Field(..., description="True if MODEL_OUTPUT_JSON complies with rules.")
    reason: Optional[str] = Field(None, description="Brief justification when incorrect.")

verifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VERIFIER_PROMPT),
        ("human", VERIFIER_DATA),
    ]
)
verifier_chain = verifier_prompt | verifier_llm.with_structured_output(Verdict)

def _parse_json_list(s: str) -> List[str]:
    """Parse and validate a JSON list of strings."""
    try:
        data = json.loads(s)
    except json.decoder.JSONDecodeError:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"Not valid JSON: {e}")
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Output is not a JSON list of strings.")
    return data

def highlight_text(text, tags):
    if not tags:
        return text
    tags = sorted([t for t in tags if "start" in t], key=lambda x: -x["start"])
    for t in tags:
        start = t["start"]
        end = t["end"]
        text = text[:start] + "**" + text[start:end] + "**" + text[end:]
    return text

def chunk_extract(raw_text: str, max_iters: int = 5):
    feedback = ""     # fed back to extractor when verifier rejects
    last_items = []   # last parsed list

    history = []

    verdicts = []

    highlighted = None

    for i in range(1, max_iters + 1):
        try:
            # 1) Extractor generates JSON list
            resp = extractor_chain.invoke({"RAW_TEXT": raw_text if not highlighted else highlighted, "feedback": feedback, "history": history})
            raw_output = resp.content.strip()

            # Ensure strict JSON list of strings
            try:
                items = _parse_json_list(raw_output)
            except ValueError as e:
                # force the extractor to return a clean JSON list next time
                feedback = (
                    "Your previous output was invalid. Return ONLY a valid JSON list of strings,"
                    " no prose, no code fences, no trailing commas."
                )
                last_items = []
                continue

            # 2) Highlight spans in the raw text (visual aid)
            tags = _result_parse(raw_text, items)[0]
            coerced_tags = coerce_tags(raw_text, [Tag.from_dict(t) for t in tags if "start" in t])
            highlighted = highlight_text(raw_text, [t.to_dict() for t in coerced_tags])

            # 3) Verifier checks correctness
            verdict: Verdict = verifier_chain.invoke({
                "HIGHLIGHTED_TEXT": highlighted,
            })

            if verdict.is_correct:
                return items, verdicts

            # 4) Not correct → feed back and retry
            feedback = (
                "Verifier rejected your last output: "
                f"{verdict.reason or 'Rule violation.'} "
                "Regenerate strictly following the spec and return ONLY a valid JSON list of strings."
            )
            last_items = items
            verdicts.append(verdict.reason)
            history.append(SystemMessage(
                content=f"Verifier feedback: {verdict.reason or 'Rule violation.'}\n"
            ))
        except Exception as e:
            print(f"Exception: {e}")
            return last_items, verdicts

    return last_items, verdicts

def agent_extract(text: str, max_iters: int = 3):
    entities = []
    for chunk in _split_text_evenly(text, 700, 200):
        chunk_entities, verdicts = chunk_extract(chunk, max_iters)
        if verdicts:
            print(verdicts)
        entities += chunk_entities
    try:
        result = [r for r in _result_parse(text, entities)[0] if "start" in r]
        result = sorted(result, key=lambda x: x['start'])
        result = set(Tag.from_dict(t) for t in result)
        return Document(result, text)
    except Exception as e:
        print(entities)
        raise e