from typing import List, Tuple, Dict
from openai import OpenAI
import json
import re
import os
from retry import retry
from functools import lru_cache
from .dataset import Tag, Document
from .constants import GPT_PROMPT
from anthropic import Anthropic


def _split_text_evenly(text, max_chunk_length=2000, min_chunk_length=1000, separators=None):
    if separators is None:
        separators = [r"\n\s*\n", r"\. ", r"\n\n", r"\n", r"\t", r", ", r" "]

    best_chunks = None
    best_score = float("inf")

    for sep in separators:
        matches = list(re.finditer(sep, text))
        if not matches:
            continue

        split_points = [m.end() for m in matches]
        split_points.append(len(text))  # ensure the end is included

        chunks = []
        prev = 0
        total_length = len(text)
        target_chunks = max(total_length // max_chunk_length, 1)
        ideal_chunk_length = total_length / target_chunks

        for p in split_points:
            current_length = p - prev

            # If current segment exceeds max, force a split here
            if current_length > max_chunk_length:
                chunks.append(text[prev:p])
                prev = p
            # If we’re near the ideal length, prefer splitting
            elif current_length >= min_chunk_length and abs(current_length - ideal_chunk_length) < ideal_chunk_length * 0.5:
                chunks.append(text[prev:p])
                prev = p
            # Otherwise, keep going to next candidate

        # Final chunk
        if prev < len(text):
            chunks.append(text[prev:])

        # Check validity
        lengths = [len(c) for c in chunks]
        if max(lengths) <= max_chunk_length and all(len(c) >= min_chunk_length for c in chunks[:-1]):
            avg = sum(lengths) / len(lengths)
            score = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            if score < best_score:
                best_chunks = chunks
                best_score = score

    return best_chunks if best_chunks else [text]

@retry(delay=1, tries=5)
def _get_openai_locations(text: str, model: str = "gpt-4o-mini") -> List[str]:
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": GPT_PROMPT
             },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    text = completion.to_dict()["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", completion.to_dict()["choices"][0]["message"]["content"].strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        raise ValueError(completion.to_dict()["choices"][0]["message"]["content"])

@retry(delay=1, tries=5)
def _get_deepseek_locations(text: str, model: str = "deepseek-chat") -> List[str]:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": GPT_PROMPT
             },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    text = completion.to_dict()["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", completion.to_dict()["choices"][0]["message"]["content"].strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        raise ValueError(completion.to_dict()["choices"][0]["message"]["content"])

@retry(delay=1, tries=5)
def _get_claude_locations(text: str, model: str = "claude-sonnet-4-5") -> List[str]:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=GPT_PROMPT,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    if len(message.content) == 0:
        return []
    text_content = message.content[0].text
    try:
        return json.loads(text_content)
    except json.decoder.JSONDecodeError:
        text_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_content.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text_content)
    except json.decoder.JSONDecodeError:
        if model == "claude-haiku-4-5":
            return []
        raise ValueError(text_content)

def _result_parse(text: str, entities: List[str]) -> Tuple[List[Dict[str, int]], int]:
    """Align detected entity strings to their character spans in the original text.

    This function receives the original `text` and an ordered list of entity
    strings (`entities`). It attempts to map each entity, in order, to a non-
    overlapping occurrence in `text` by finding character start/end indices.

    It uses dynamic programming to explore two choices for each entity:
      1) Match it at the next valid occurrence after the current index.
      2) Skip it (counted as one error) if no suitable occurrence should be used.

    The DP minimizes the number of skipped entities (errors). For matched
    entities, the function returns dictionaries with keys: `text`, `start`,
    and `end`. For skipped ones, only `text` is returned. Consumers typically
    filter out the skipped entries later.

    Returns a tuple of:
      - a list of entity dicts (some with spans, some without when skipped)
      - the total number of skipped entities (minimal under the constraints)
    """

    @lru_cache(None)
    def dp(i: int, idx: int) -> Tuple[int, List[Dict[str, int]]]:
        # Base case: processed all entities — no further errors, no more spans
        if i == len(entities):
            return 0, []

        entity = entities[i]
        min_errors = float('inf')
        best_result = []

        # Try to match current entity at every valid occurrence from `idx` onward
        start_idx = idx
        while True:
            # Word-boundary-like match: avoid partial matches within words
            m = re.search(rf"(?<![A-Za-z]){re.escape(entity)}(?![A-Za-z])", text[start_idx:])
            start = m.start() + start_idx if m else -1
            if start == -1:
                break

            end = start + len(entity)
            # Recurse to place the next entities after this span
            next_errors, next_result = dp(i + 1, end)

            if next_errors < min_errors:
                min_errors = next_errors
                best_result = [{'text': entity, 'start': start, 'end': end}] + next_result

            # Continue searching for a later occurrence to see if it yields fewer skips
            start_idx = start + 1

        # Option: skip this entity (incurs one error) and keep looking from the same index
        next_errors, next_result = dp(i + 1, idx)
        next_errors += 1

        if next_errors < min_errors:
            min_errors = next_errors
            best_result = [{'text': entity}] + next_result

        return min_errors, best_result

    # Start DP from the first entity and the beginning of the text
    error_count, result = dp(0, 0)
    return result, error_count

def coerce_tags(text: str, tags: List[Tag]):
    pattern = r'^\s*(, | and | in |\s|-)\s*$'

    sorted_tags = sorted(tags, key=lambda x: x.start)
    new_tags = []
    for i in range(len(sorted_tags)):
        curr = sorted_tags[i]
        if not new_tags:
            new_tags.append(curr)
            continue
        prev = new_tags[-1]
        # Check if tags are adjacent (no gap or only whitespace) or match the pattern
        gap_text = text[prev.end:curr.start]
        is_adjacent = prev.end == curr.start or (gap_text.strip() == '' and len(gap_text) <= 1)
        if is_adjacent or re.match(pattern, gap_text, flags=re.IGNORECASE):
            prev = new_tags.pop(-1)
            new_tags.append(Tag(text[prev.start:curr.end], prev.start, curr.end))
        else:
            new_tags.append(curr)
    return new_tags

def tag_text(text: str, model: str = "gpt-4o-mini", max_chunk_length=2000, min_chunk_length=1000) -> Document:
    entities = []
    if "claude" in model.lower():
        get_locations = _get_claude_locations
    elif "deepseek" in model.lower():
        get_locations = _get_deepseek_locations
    else:
        get_locations = _get_openai_locations
    for chunk in _split_text_evenly(text, max_chunk_length=max_chunk_length, min_chunk_length=min_chunk_length):
        entities += get_locations(chunk, model=model)
    result = [r for r in _result_parse(text, entities)[0] if "start" in r]
    result = sorted(result, key=lambda x: x['start'])
    result = set(Tag.from_dict(t) for t in result)
    return Document(result, text)
