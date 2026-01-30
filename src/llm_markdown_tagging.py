from typing import List, Tuple, Dict
from openai import OpenAI
import re
import os
from retry import retry
from functools import lru_cache
from .dataset import Tag, Document
from .constants import GPT_MARKDOWN_PROMPT
from anthropic import Anthropic


TAG_RE = re.compile(r'@@(.*?)##', flags=re.DOTALL)


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

def _openai_result_parse(text: str, entities: List[str]) -> Tuple[List[Dict[str, int]], int]:
    @lru_cache(None)
    def dp(i: int, idx: int) -> Tuple[int, List[Dict[str, int]]]:
        if i == len(entities):
            return 0, []

        entity = entities[i]
        min_errors = float('inf')
        best_result = []

        start_idx = idx
        while True:
            m = re.search(rf"(?<![A-Za-z]){re.escape(entity)}(?![A-Za-z])", text[start_idx:])
            start = m.start() + start_idx if m else -1
            if start == -1:
                break

            end = start + len(entity)
            next_errors, next_result = dp(i + 1, end)

            if next_errors < min_errors:
                min_errors = next_errors
                best_result = [{'text': entity, 'start': start, 'end': end}] + next_result

            start_idx = start + 1

        next_errors, next_result = dp(i + 1, idx)
        next_errors += 1

        if next_errors < min_errors:
            min_errors = next_errors
            best_result = [{'text': entity}] + next_result

        return min_errors, best_result

    error_count, result = dp(0, 0)
    return result, error_count

@retry(delay=1, tries=5)
def _get_openai_locations(text: str, model: str = "gpt-4o-mini") -> Tuple[List[Dict[str, int]], int]:
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": GPT_MARKDOWN_PROMPT
             },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    tagged = completion.to_dict()["choices"][0]["message"]["content"]
    text_wout_tags= re.sub(TAG_RE, r'\1', tagged)
    if text_wout_tags.rstrip().lstrip() == text.rstrip().lstrip():
        return _extract_entities_and_spans(tagged, text)
    return _openai_result_parse(text, [m.group(1) for m in TAG_RE.finditer(tagged)])

@retry(delay=1, tries=5)
def _get_deepseek_locations(text: str, model: str = "deepseek-chat") -> Tuple[List[Dict[str, int]], int]:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": GPT_MARKDOWN_PROMPT
             },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    tagged = completion.to_dict()["choices"][0]["message"]["content"]
    text_wout_tags= re.sub(TAG_RE, r'\1', tagged)
    if text_wout_tags.rstrip().lstrip() == text.rstrip().lstrip():
        return _extract_entities_and_spans(tagged, text)
    return _openai_result_parse(text, [m.group(1) for m in TAG_RE.finditer(tagged)])

@retry(delay=1, tries=5)
def _get_claude_locations(text: str, model: str = "claude-sonnet-4-5") -> Tuple[List[Dict[str, int]], int]:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=GPT_MARKDOWN_PROMPT,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    if len(message.content) == 0:
        return [], 0
    tagged = message.content[0].text
    text_wout_tags = re.sub(TAG_RE, r'\1', tagged)
    if text_wout_tags.rstrip().lstrip() == text.rstrip().lstrip():
        return _extract_entities_and_spans(tagged, text)
    return _openai_result_parse(text, [m.group(1) for m in TAG_RE.finditer(tagged)])

def leading_space_diff(text_wout_tags: str, text: str) -> int:
    ws1 = len(text_wout_tags) - len(text_wout_tags.lstrip())
    ws2 = len(text) - len(text.lstrip())
    return ws1 - ws2

def _extract_entities_and_spans(tagged: str, original_text: str) -> Tuple[List[Dict[str, int]], int]:
    extra = leading_space_diff(re.sub(TAG_RE, r'\1', tagged), original_text)
    n = len(tagged)

    idx_map = [0] * (n + 1)
    clean_chars = []
    clean_i = 0
    i = 0
    while i < n:
        idx_map[i] = clean_i
        if tagged.startswith('@@', i) or tagged.startswith('##', i):
            i += 2
            continue
        clean_chars.append(tagged[i])
        clean_i += 1
        i += 1
    idx_map[n] = clean_i

    entities = []
    for m in TAG_RE.finditer(tagged):
        ent = m.group(1)
        start_in_tagged = m.start(1)  # first char after '@@'
        start_in_clean = idx_map[start_in_tagged]
        end_in_clean = start_in_clean + len(ent)
        entities.append({"text": ent, "start": start_in_clean+extra, "end": end_in_clean+extra})

    return entities, 0

def markdown_tag_text(text: str, model: str = "gpt-4o-mini") -> Document:
    tags = []
    chunk_total = 0
    if "claude" in model.lower():
        get_locations = _get_claude_locations
    elif "deepseek" in model.lower():
        get_locations = _get_deepseek_locations
    else:
        get_locations = _get_openai_locations
    for chunk in _split_text_evenly(text, 500, 200):
        chunk_tags, e = get_locations(chunk, model=model)
        tags += [{"text": t['text'],
                  "start": t["start"]+chunk_total,
                  "end": t["end"]+chunk_total} for t in chunk_tags if "start" in t]
        chunk_total += len(chunk)
    tags = sorted(tags, key=lambda x: x['start'])
    tags = set(Tag.from_dict(t) for t in tags)
    return Document(tags, text)
