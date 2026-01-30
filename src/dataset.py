from typing import Set, Dict, Generator
from dataclasses import dataclass

@dataclass(frozen=True)
class Tag:
    text: str
    start: int
    end: int

    def __post_init__(self):
        if not len(self.text) == self.end - self.start:
            raise ValueError(f"Text '{self.text}' with length {len(self.text)} does not match the positions {self.end}-{self.start}")

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d):
        return cls(d["text"], d["start"], d["end"])
    
    def __hash__(self):
        return hash((self.text, self.start, self.end))
    
    def __repr__(self):
        return f"Tag(text={self.text}, start={self.start}, end={self.end})"

    def __len__(self):
        return self.end - self.start

@dataclass(frozen=True)
class Document:
    tags: Set[Tag]
    text: str

    def __post_init__(self):
        last_end = -1
        for tag in sorted(self.tags, key=lambda x: x.start):
            if tag.start < last_end:
                raise ValueError("There are repeated tags")
            last_end = tag.end
        for tag in self.tags:
            assert tag.text == self.text[tag.start: tag.end]

    def to_dict(self):
        return {"text": self.text, "tags": [t.to_dict() for t in self.tags]}

    @classmethod
    def from_dict(cls, d, tags_name="tags"):
        return cls(set(Tag.from_dict(t) for t in d[tags_name]), d["text"])

    def __len__(self):
        return len(self.tags)

    def __contains__(self, tag):
        return tag in self.tags

    @property
    def unique_tags(self):
        sorted_tags = sorted(self.tags, key=lambda x: x.start)
        amount = 0
        prev_end = -1
        for tag in sorted_tags:
            if tag.start > prev_end:
                amount += 1
            prev_end = tag.end
        return amount

@dataclass(frozen=True)
class Dataset:
    documents: Dict[int, Document]
    
    def to_dict(self):
        return {k: v.to_dict() for k,v in self.documents.items()}

    @classmethod
    def from_dict(cls, d, tags_name="tags"):
         return Dataset({k: Document.from_dict(v, tags_name) for k, v in d.items()})

    def __len__(self):
        return len(self.documents)

    def keys(self) -> Generator[int, None, None]:
        for k in self.documents.keys():
            yield k

    def __getitem__(self, k) -> Document:
        return self.documents[k]

    def __contains__(self, k):
        return k in self.documents