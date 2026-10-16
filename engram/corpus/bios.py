"""Synthetic bioS-style biographies with exact probes (slice D; research/phase1-corpus.md §c).

The design follows bioS (Allen-Zhu & Li, arXiv:2309.14316): each fictitious person gets
independently sampled attributes (birth date, birth city, university, major, employer, employer
city), written as one sentence per attribute from a pool of templates, with the person's sentences
after the first permuted. Engram additions, matching engram-bench's categories:

* **updates**: a later paragraph changes a person's employer and work city; probes asked after
  both are written expect the new value and reject the old one (stale-value-wrong, as in FAMA);
* **abstention**: optional attributes (instrument, mentor) that were never stated;
* **indirect**: "Who should I ask about the leaking roof?" answered by the person whose expertise
  is roofing.

Gold answers come straight from the structured record; nothing here calls a model. Name pools,
value lists and templates are disjoint from engram-bench (``engram/bench/world.py``);
``check_disjoint`` asserts it, and the build calls it.

Each document is one "world" of people. Probes are stored at document level with character
spans (``facts``: every statement of the value, in order; ``anchor``: where the person is
introduced, for abstention), so the episode builder can work out which values a memory saw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

SOURCE = "engram_bios"
LICENCE = "Engram-generated (Apache-2.0)"

# Disjoint from engram/bench/world.py by construction (checked by check_disjoint).
_FIRST_A = [
    "bar",
    "che",
    "dun",
    "gim",
    "hal",
    "jor",
    "kes",
    "lum",
    "mav",
    "nir",
    "pel",
    "rus",
    "tav",
    "wen",
    "yor",
    "fos",
    "grib",
    "osk",
]
_FIRST_B = [
    "ett",
    "ina",
    "ard",
    "ula",
    "ric",
    "ova",
    "emi",
    "on",
    "ith",
    "aly",
    "umn",
    "esh",
    "orin",
    "aya",
]
_LAST_A = [
    "hask",
    "mull",
    "pern",
    "quist",
    "rath",
    "stov",
    "tamb",
    "vend",
    "wick",
    "yarr",
    "brom",
    "corv",
    "dell",
    "fenn",
    "garr",
]
_LAST_B = ["ell", "orn", "am", "ins", "ard", "ow", "et", "ick", "um", "en"]
_LAST_END = ["field", "holt", "mere", "stead", "vale", "croft", "by", "ham", "more", "burn"]
_PLACE_A = [
    "ost",
    "bel",
    "car",
    "dor",
    "eth",
    "fal",
    "glen",
    "har",
    "ins",
    "kel",
    "mar",
    "nor",
    "pol",
    "ros",
    "sel",
    "tur",
    "ush",
    "wyn",
]
_PLACE_B = ["an", "ev", "ir", "ol", "ur", "ast", "em", "ith", "ond", "y"]
_PLACE_END = ["port", "ford", "burgh", "mouth", "dale", "minster", "haven", "bridge", "gate"]
_EMPLOYER_KIND = [
    "Holdings",
    "Cooperative",
    "Foundry",
    "Partners",
    "Guild",
    "Outfitters",
    "Instruments",
    "Textiles",
    "Freight",
    "Mills",
    "Shipyards",
    "Assurance",
]
_MAJORS = [
    "marine biology",
    "civil engineering",
    "medieval history",
    "applied mathematics",
    "organic chemistry",
    "comparative literature",
    "structural geology",
    "linguistics",
    "astrophysics",
    "agricultural economics",
    "naval architecture",
    "musicology",
    "forestry",
    "materials science",
    "epidemiology",
    "cartography",
    "classics",
    "industrial design",
    "hydrology",
    "philosophy of science",
]
_INSTRUMENTS = [
    "the oboe",
    "the cello",
    "the bassoon",
    "the harp",
    "the accordion",
    "the french horn",
    "the viola",
    "the mandolin",
    "the marimba",
    "the bagpipes",
]
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
# (expertise as stated, the need a probe asks about)
_EXPERTISE = [
    ("tax law", "the tax question"),
    ("roof repair", "the leaking roof"),
    ("orchid care", "the wilting orchids"),
    ("contract negotiation", "the supplier contract"),
    ("bicycle maintenance", "the broken bicycle"),
    ("immigration paperwork", "the visa forms"),
    ("piano tuning", "the out-of-tune piano"),
    ("bookkeeping", "the messy accounts"),
    ("beekeeping", "the swarm in the garden"),
    ("dog training", "the puppy that keeps barking"),
    ("wedding photography", "the wedding photos"),
    ("sourdough baking", "the flat bread loaves"),
    ("chimney sweeping", "the smoky fireplace"),
    ("translation from Portuguese", "the letter in Portuguese"),
    ("vintage watch repair", "the stopped pocket watch"),
    ("kitchen plumbing", "the blocked sink"),
    ("sailing instruction", "the sailing lessons"),
    ("furniture restoration", "the scratched oak table"),
    ("grant writing", "the funding application"),
    ("tile setting", "the cracked bathroom tiles"),
    ("violin making", "the cracked violin"),
    ("tree surgery", "the dying elm"),
    ("speech coaching", "the conference talk"),
    ("data recovery", "the dead hard drive"),
]
_PRONOUNS = {"he": ("He", "his"), "she": ("She", "her")}

# Templates: {S} subject, {P} possessive. Several phrasings per attribute (bioS uses ~50).
T_BIRTH_DATE = [
    "{S} was born on {v}.",
    "{S} came into the world on {v}.",
    "{S} celebrates a birthday on {d}, having been born on {v}.",
    "The birth of {S} took place on {v}.",
    "{S} entered life on {v}.",
    "Records show that {S} was born on {v}.",
]
T_BIRTH_CITY = [
    "{S} was born in {v}.",
    "{S} spent {P} early years in {v}, where {P} family lived.",
    "{S} is a native of {v}.",
    "{S} grew up in the town of {v}.",
    "The hometown of {S} is {v}.",
    "{S} first saw daylight in {v}.",
]
T_UNIVERSITY = [
    "{S} studied at {v}.",
    "{S} received {P} degree from {v}.",
    "{S} is a graduate of {v}.",
    "{S} enrolled at {v} after school.",
    "{S} spent four years at {v}.",
    "{S} graduated from {v}.",
]
T_MAJOR = [
    "{S} majored in {v}.",
    "{S} focused {P} studies on {v}.",
    "{S} chose {v} as a field.",
    "{S} earned a degree in {v}.",
    "The subject {S} studied was {v}.",
    "{S} specialised in {v} at university.",
]
T_EMPLOYER = [
    "{S} works for {v}.",
    "{S} is employed by {v}.",
    "{S} has a job at {v}.",
    "{S} joined the staff of {v}.",
    "{S} earns a living at {v}.",
    "{S} took a position at {v}.",
]
T_EMPLOYER_CITY = [
    "{S} works in {v}.",
    "The office of {S} is in {v}.",
    "{S} commutes to {v} for work.",
    "{S} is based in {v} for work.",
    "Each workday, {S} goes to {v}.",
]
T_INSTRUMENT = [
    "{S} plays {v}.",
    "In the evenings {S} practises {v}.",
    "{S} is a keen player of {v}.",
    "{S} learned to play {v} as a hobby.",
]
T_MENTOR = [
    "{S} was mentored by {v}.",
    "{S} credits {v} as a mentor.",
    "{S} trained under {v}.",
    "The mentor of {S} was {v}.",
]
T_EXPERTISE = [
    "{S} is the person to ask about {v}.",
    "{S} knows more about {v} than anyone else here.",
    "Friends turn to {S} for help with {v}.",
    "{S} has years of experience in {v}.",
]
T_UPDATE = [
    "In {y}, {S} left {old} and joined {v} in {city}.",
    "Some years later, {S} moved from {old} to a new job at {v}, based in {city}.",
    "By {y}, {S} no longer worked for {old}; {P} new employer was {v}, in {city}.",
]

QUESTIONS = {
    "birth_date": "When was {name} born?",
    "birth_city": "Where was {name} born?",
    "university": "Which university did {name} attend?",
    "major": "What did {name} study?",
    "employer": "Which company does {name} work for?",
    "employer_city": "In which city does {name} work?",
    "instrument": "Which instrument does {name} play?",
    "mentor": "Who mentored {name}?",
}
CORE = ["birth_date", "birth_city", "university", "major", "employer", "employer_city"]
EXTRAS = ["instrument", "mentor"]
TEMPLATES = {
    "birth_date": T_BIRTH_DATE,
    "birth_city": T_BIRTH_CITY,
    "university": T_UNIVERSITY,
    "major": T_MAJOR,
    "employer": T_EMPLOYER,
    "employer_city": T_EMPLOYER_CITY,
    "instrument": T_INSTRUMENT,
    "mentor": T_MENTOR,
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def first_name(rng: random.Random) -> str:
    return _cap(rng.choice(_FIRST_A) + rng.choice(_FIRST_B))


def surname(rng: random.Random) -> str:
    return _cap(rng.choice(_LAST_A) + rng.choice(_LAST_B) + rng.choice(_LAST_END))


def place(rng: random.Random) -> str:
    return _cap(rng.choice(_PLACE_A) + rng.choice(_PLACE_B) + rng.choice(_PLACE_END))


def employer(rng: random.Random) -> str:
    return f"{_cap(rng.choice(_LAST_A) + rng.choice(_PLACE_B))} {rng.choice(_EMPLOYER_KIND)}"


def university(rng: random.Random) -> str:
    style = rng.randrange(3)
    if style == 0:
        return f"the University of {place(rng)}"
    if style == 1:
        return f"{place(rng)} College"
    return f"the {place(rng)} Institute of Technology"


@dataclass
class Person:
    first: str
    last: str
    pronoun: str
    attrs: dict[str, str] = field(default_factory=dict)
    expertise: tuple[str, str] | None = None
    update: dict[str, str] | None = None  # {"employer", "employer_city", "year"}

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


def sample_person(rng: random.Random, used: set[str], p_extra: float) -> Person:
    while True:
        p = Person(first_name(rng), surname(rng), rng.choice(list(_PRONOUNS)))
        if p.name not in used:
            used.add(p.name)
            break
    year = rng.randint(1931, 2004)
    p.attrs["birth_date"] = f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}, {year}"
    p.attrs["birth_city"] = place(rng)
    p.attrs["university"] = university(rng)
    p.attrs["major"] = rng.choice(_MAJORS)
    p.attrs["employer"] = employer(rng)
    p.attrs["employer_city"] = place(rng)
    if rng.random() < p_extra:
        p.attrs["instrument"] = rng.choice(_INSTRUMENTS)
    if rng.random() < p_extra:
        p.attrs["mentor"] = f"{first_name(rng)} {surname(rng)}"
    return p


class _Doc:
    """Accumulates paragraphs and remembers the character span of every sentence."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0

    def paragraph(self, sentences: list[str]) -> list[tuple[int, int]]:
        if self.parts:
            self.parts.append("\n\n")
            self.length += 2
        spans = []
        for i, s in enumerate(sentences):
            if i:
                self.parts.append(" ")
                self.length += 1
            spans.append((self.length, self.length + len(s)))
            self.parts.append(s)
            self.length += len(s)
        return spans

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _render(tmpl: str, p: Person, subject: str, **kw: str) -> str:
    return _cap(tmpl.format(S=subject, P=_PRONOUNS[p.pronoun][1], **kw))


def generate_doc(
    seed: int, index: int, *, people: int = 40, p_update: float = 0.3, p_extra: float = 0.5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One deterministic biography document and its probes."""
    rng = random.Random(f"engram-bios:{seed}:{index}")
    doc_id = f"bios-s{seed}-{index:06d}"
    used: set[str] = set()
    persons = [sample_person(rng, used, p_extra) for _ in range(people)]
    expertise = rng.sample(_EXPERTISE, min(len(_EXPERTISE), max(1, people // 3)))
    for p, ex in zip(rng.sample(persons, len(expertise)), expertise, strict=True):
        p.expertise = ex

    doc = _Doc()
    facts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    anchors: dict[str, tuple[int, int]] = {}
    pending_updates: list[tuple[int, Person]] = []

    def add_fact(p: Person, attr: str, value: str, span: tuple[int, int]) -> None:
        facts.setdefault((p.name, attr), []).append(
            {"value": value, "start": span[0], "end": span[1]}
        )

    def write_update(p: Person) -> None:
        old = p.attrs["employer"]
        new = employer(rng)
        while new == old:
            new = employer(rng)
        city = place(rng)
        year = str(int(p.attrs["birth_date"][-4:]) + rng.randint(25, 45))
        subject = p.name if rng.random() < 0.7 else p.first
        tmpl = rng.choice(T_UPDATE)
        sentence = _render(tmpl, p, subject, y=year, old=old, v=new, city=city)
        (span,) = doc.paragraph([sentence])
        add_fact(p, "employer", new, span)
        add_fact(p, "employer_city", city, span)
        p.update = {"employer": new, "employer_city": city, "year": year}

    for n, p in enumerate(persons):
        stated = [a for a in CORE + EXTRAS if a in p.attrs]
        rest = stated[1:]
        rng.shuffle(rest)
        order = [stated[0], *rest]
        sentences, keys = [], []
        for i, attr in enumerate(order):
            tmpl = rng.choice(TEMPLATES[attr])
            if i == 0:
                subject = p.name
            elif tmpl.startswith("{S}"):
                subject = rng.choice([_PRONOUNS[p.pronoun][0], p.first, p.name])
            else:  # a pronoun only reads right at the start of a sentence
                subject = rng.choice([p.first, p.name])
            v = p.attrs[attr]
            extra = {"d": v.rsplit(",", 1)[0]} if attr == "birth_date" else {}
            sentences.append(_render(tmpl, p, subject, v=v, **extra))
            keys.append(attr)
        if p.expertise:
            # full name: the indirect probe's gold must be stated in its evidence sentence
            sentences.append(_render(rng.choice(T_EXPERTISE), p, p.name, v=p.expertise[0]))
            keys.append("expertise")
        spans = doc.paragraph(sentences)
        anchors[p.name] = spans[0]
        for attr, span in zip(keys, spans, strict=True):
            value = p.name if attr == "expertise" else p.attrs[attr]
            add_fact(p, attr, value, span)
        if rng.random() < p_update:
            pending_updates.append((n + rng.randint(2, 8), p))
        due = [u for u in pending_updates if u[0] <= n]
        for u in due:
            pending_updates.remove(u)
            write_update(u[1])
    for _, p in pending_updates:
        write_update(p)

    text = doc.text
    probes: list[dict[str, Any]] = []
    for p in persons:
        for attr in CORE + EXTRAS:
            q = QUESTIONS[attr].format(name=p.name)
            pid = f"{doc_id}-{len(probes)}"
            if attr in p.attrs:
                fs = facts[(p.name, attr)]
                kind = "update" if len(fs) > 1 else "fact"
                probes.append(_probe(pid, doc_id, q, kind, attr, fs, None))
            else:
                probes.append(_probe(pid, doc_id, q, "abstain", attr, [], anchors[p.name]))
        if p.expertise:
            fs = facts[(p.name, "expertise")]
            q = f"Who should I ask about {p.expertise[1]}?"
            probes.append(
                _probe(
                    f"{doc_id}-{len(probes)}",
                    doc_id,
                    q,
                    "indirect",
                    "expertise",
                    fs,
                    None,
                    accept=[p.name],
                )
            )
    for pr in probes:
        for f in pr["facts"]:
            assert f["value"] in text[f["start"] : f["end"]], (pr, f)
    record = {
        "source": SOURCE,
        "doc_id": doc_id,
        "text": text,
        "licence": LICENCE,
        "meta": {"seed": seed, "index": index, "people": people, "generator": "bios-v1"},
    }
    return record, probes


def _probe(
    pid: str,
    doc_id: str,
    question: str,
    kind: str,
    attr: str,
    facts: list[dict[str, Any]],
    anchor: tuple[int, int] | None,
    accept: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": pid,
        "source": SOURCE,
        "doc_id": doc_id,
        "question": question,
        "type": kind,
        "attribute": attr,
        "facts": facts,
        "anchor": list(anchor) if anchor else None,
        "accept": accept or [],
        "generator": "template",
    }


def generate(
    seed: int, n_docs: int, *, people: int = 40, p_update: float = 0.3, p_extra: float = 0.5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs, probes = [], []
    for i in range(n_docs):
        d, ps = generate_doc(seed, i, people=people, p_update=p_update, p_extra=p_extra)
        docs.append(d)
        probes.extend(ps)
    return docs, probes


def bench_name_pools() -> tuple[set[str], set[str]]:
    """Every first name and surname engram-bench's generator can produce (lowercased)."""
    from engram.bench import world

    firsts = {(a + b).lower() for a in world._SYL_A for b in world._SYL_B}
    lasts = {
        (a + b + e).lower() for a in world._SYL_A for b in world._SYL_B for e in world._SURNAME_END
    }
    return firsts, lasts


def bench_values(draws: int = 400) -> set[str]:
    """Values engram-bench plants (sampled from each slot's generator), lowercased."""
    from engram.bench import world

    rng = random.Random(0)
    return {slot.values(rng).lower() for slot in world.SLOTS for _ in range(draws)}


def check_disjoint(docs: list[dict[str, Any]], probes: list[dict[str, Any]]) -> list[str]:
    """Problems if any bio name or value could come from engram-bench (empty list = disjoint)."""
    firsts, lasts = bench_name_pools()
    values = bench_values()
    problems = []
    vocab_first = {(a + b) for a in _FIRST_A for b in _FIRST_B}
    vocab_last = {(a + b + e) for a in _LAST_A for b in _LAST_B for e in _LAST_END}
    vocab_first |= {(a + b) for a in _LAST_A for b in _PLACE_B}  # employer stems
    for name in sorted(vocab_first & (firsts | lasts)):
        problems.append(f"first-name pool overlaps engram-bench: {name}")
    for name in sorted(vocab_last & (firsts | lasts)):
        problems.append(f"surname pool overlaps engram-bench: {name}")
    for pr in probes:
        for f in pr["facts"]:
            if f["value"].lower() in values:
                problems.append(f"{pr['id']}: value {f['value']!r} is an engram-bench value")
    return problems
