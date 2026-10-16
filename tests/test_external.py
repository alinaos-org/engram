import json
from pathlib import Path

import pytest

from engram.bench import dataset, external
from engram.bench.runner import run
from engram.bench.schema import Category, Phase
from engram.systems.baselines import OracleMemory

FIXTURES = Path(__file__).parent / "fixtures"
LME = FIXTURES / "longmemeval_tiny.json"
LOCOMO = FIXTURES / "locomo_tiny.json"
REAL = Path(__file__).parent.parent / "data" / "external"


class Recorder:
    """A memory system that only records the order in which the runner drives it."""

    name = "recorder"

    def __init__(self) -> None:
        self.log: list[tuple] = []
        self.session = 0
        self.writing = True

    def load_user(self, user_id: str) -> None:
        self.log.append(("load", user_id))
        self.session = 0

    def unload_user(self) -> None:
        self.log.append(("unload",))

    def start_session(self, *, write: bool = True) -> None:
        self.writing = write

    def respond(self, message: str) -> str:
        self.log.append(("turn" if self.writing else "probe", self.session, message))
        return "I don't know."

    def end_session(self) -> None:
        if self.writing:
            self.session += 1

    def sleep(self) -> None:
        self.log.append(("sleep", self.session - 1))

    def delete_user(self, user_id: str) -> None:
        pass

    def describe(self) -> dict:
        return {"system": "recorder"}


@pytest.fixture
def lme():
    return external.load_longmemeval(LME)


@pytest.fixture
def locomo():
    return external.load_locomo(LOCOMO)


def _valid(bench) -> None:
    sleeps = set(bench.sleeps)
    assert bench.config.users == len(bench.users)
    assert bench.config.sessions == max(len(u.sessions) for u in bench.users)
    ids = [p.id for p in bench.probes()]
    assert len(ids) == len(set(ids))
    for user in bench.users:
        assert [s.index for s in user.sessions] == list(range(len(user.sessions)))
        assert all(s.user_id == user.id and not s.incognito for s in user.sessions)
        days = [s.day for s in user.sessions]
        assert days == sorted(days) and days[0] == 0
        last = len(user.sessions) - 1
        for p in user.probes:
            assert p.user_id == user.id
            assert p.ask_after_session == last
            assert p.phase == (Phase.POST_SLEEP if last in sleeps else Phase.PRE_SLEEP)
            assert (p.gold is None) == (p.category == Category.ABSTENTION)
            if p.gold is not None:
                assert p.gold in p.accept
            else:
                assert not p.accept
    assert bench.needs_judge <= set(ids)


def test_longmemeval_conversion(lme):
    _valid(lme)
    assert [u.id for u in lme.users] == ["lme-ku001", "lme-tr002_abs"]
    assert lme.writer == "longmemeval:longmemeval_tiny.json"
    ku, ab = lme.users

    # Sessions are put in date order (the oracle file is unsorted), and dated on their first turn.
    firsts = [s.turns[0].text for s in ku.sessions]
    assert firsts[0].startswith("[Conversation date: 2023/05/02 (Tue) 10:15]\nI live in Denver")
    assert firsts[1].startswith("[Conversation date: 2023/05/20 (Sat) 08:05]\n")
    assert "moving to Portland" in firsts[2]
    assert [s.day for s in ku.sessions] == [0, 18, 18]

    # Assistant turns are kept with a prefix; empty messages are dropped.
    texts = [t.text for t in ku.sessions[0].turns]
    assert len(texts) == 4
    assert (
        texts[1]
        == external.ASSISTANT_PREFIX
        + "Try the Mount Falcon trail, about thirty minutes from Denver."
    )
    assert external.ASSISTANT_PREFIX + "Welcome back!" in ku.sessions[1].turns[0].text
    assert len(ku.sessions[1].turns) == 2
    # Unicode line separators become newlines (dataset.load splits users.jsonl with splitlines).
    assert "banana bread?\nSomething quick." in ku.sessions[1].turns[1].text

    (p,) = ku.probes
    assert p.category == Category.UPDATE
    assert p.question == "[Current date: 2023/06/01 (Thu) 09:00] Which city do I live in now?"
    assert (p.gold, p.accept) == ("Portland", ["Portland"])
    assert (p.ask_after_session, p.phase, p.timing) == (2, Phase.POST_SLEEP, "b")

    (q,) = ab.probes
    assert q.category == Category.ABSTENTION and q.gold is None
    assert (q.ask_after_session, q.phase, q.timing) == (1, Phase.PRE_SLEEP, "a")
    assert lme.needs_judge == set()


def test_longmemeval_options(tmp_path):
    assert len(external.load_longmemeval(LME, limit=1).users) == 1
    user_only = external.load_longmemeval(LME, assistant_turns=False)
    turns = [t.text for u in user_only.users for s in u.sessions for t in s.turns]
    assert turns and not any(external.ASSISTANT_PREFIX in t for t in turns)

    data = json.loads(LME.read_text())
    data[0]["question_type"] = "single-session-preference"
    data[0]["answer"] = "The user would prefer suggestions for hikes near Portland."
    data[1]["question_id"] = "tr002"
    data[1]["answer"] = 3
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data))
    bench = external.load_longmemeval(path)
    pref, temporal = (u.probes[0] for u in bench.users)
    assert pref.category == Category.PREFERENCE
    assert temporal.category == Category.TEMPORAL
    assert (temporal.gold, temporal.accept) == ("3", ["3", "three"])
    # Rubric answers and temporal answers can only be judged, not string-matched.
    assert bench.needs_judge == {pref.id, temporal.id}


def test_longmemeval_categories():
    cat = external.longmemeval_category
    assert cat({"question_id": "a", "question_type": "knowledge-update"}) == Category.UPDATE
    assert cat({"question_id": "a", "question_type": "temporal-reasoning"}) == Category.TEMPORAL
    assert cat({"question_id": "a", "question_type": "multi-session"}) == Category.FACT_RECALL
    assert cat({"question_id": "a", "question_type": "single-session-assistant"}) == (
        Category.FACT_RECALL
    )
    assert cat({"question_id": "a_abs", "question_type": "knowledge-update"}) == (
        Category.ABSTENTION
    )


def test_locomo_conversion(locomo):
    _valid(locomo)
    assert [u.id for u in locomo.users] == ["conv-t1", "conv-t2"]
    t1, t2 = locomo.users
    assert t1.name == "Ana and Ben"
    # session_4_date_time has no session_4: it is ignored.
    assert len(t1.sessions) == 3 and len(t2.sessions) == 2
    assert [s.day for s in t1.sessions] == [0, 50, 56]
    first = t1.sessions[0].turns
    assert first[0].text == (
        "[Conversation date: 1:56 pm on 8 May, 2023]\nAna: Hi Ben! I moved to Lisbon yesterday."
    )
    assert first[1].text.startswith("Ben: ")
    assert first[2].text.endswith("[shares a photo: a photo of a balcony with red flowers]")

    by_q = {p.question: p for p in t1.probes}
    assert len(t1.probes) == 5 and len(t2.probes) == 2
    assert by_q["Which city did Ana move to?"].category == Category.FACT_RECALL
    assert by_q["When did Ana move to Lisbon?"].category == Category.TEMPORAL
    count = by_q["How many times did Ana go to the beach that week?"]
    assert (count.gold, count.accept) == ("2", ["2", "two"])
    adv = by_q["What is the name of Ana's beagle?"]
    assert adv.category == Category.ABSTENTION and adv.gold is None
    assert adv.reject == ["Rufus"]
    assert all(p.phase == Phase.POST_SLEEP for p in t1.probes)
    assert all(p.phase == Phase.PRE_SLEEP for p in t2.probes)
    assert locomo.needs_judge == {
        by_q["When did Ana move to Lisbon?"].id,
        by_q["Would Ben enjoy a dog-friendly beach holiday?"].id,
    }


def test_summarize(lme, locomo):
    assert external.summarize(lme) == {"update": 1, "abstention": 1}
    assert external.summarize(locomo) == {"fact_recall": 4, "temporal": 1, "abstention": 2}
    assert external.judge_ids(locomo) == locomo.needs_judge
    assert external.judge_ids(dataset.Benchmark(dataset.BenchConfig())) == set()


def test_accept_variants():
    assert external.accept_variants("Portland") == ["Portland"]
    assert external.accept_variants("12") == ["12", "twelve"]
    assert external.accept_variants("2022") == ["2022"]
    assert external.accept_variants("25 minutes and 50 seconds (or 25:50)") == [
        "25 minutes and 50 seconds (or 25:50)",
        "25 minutes and 50 seconds",
        "25:50",
    ]


@pytest.mark.parametrize("loader", ["lme", "locomo"])
def test_oracle_scores_perfectly(loader, request):
    bench = request.getfixturevalue(loader)
    result = run(bench, OracleMemory(bench), leakage=False)
    assert len(result.scores) == sum(len(u.probes) for u in bench.users)
    assert all(s.correct for s in result.scores), [s for s in result.scores if not s.correct]


@pytest.mark.parametrize("loader", ["lme", "locomo"])
def test_probes_asked_after_last_session(loader, request):
    bench = request.getfixturevalue(loader)
    system = Recorder()
    run(bench, system, leakage=False)
    sleeps = set(bench.sleeps)
    for user in bench.users:
        start = system.log.index(("load", user.id))
        end = system.log.index(("unload",), start)
        log = system.log[start + 1 : end]
        turns = [e for e in log if e[0] == "turn"]
        assert len(turns) == sum(len(s.turns) for s in user.sessions)
        last = len(user.sessions) - 1
        first_probe = next(i for i, e in enumerate(log) if e[0] == "probe")
        # Every scripted turn is played before any probe, and probes come after the final sleep.
        assert all(e[0] != "turn" for e in log[first_probe:])
        assert [e[2] for e in log if e[0] == "probe"] == [p.question for p in user.probes]
        sleeps_before = [e for e in log[:first_probe] if e[0] == "sleep"]
        assert [e[1] for e in sleeps_before] == [s for s in sorted(sleeps) if s <= last]


@pytest.mark.parametrize("loader", ["lme", "locomo"])
def test_save_load_roundtrip(loader, request, tmp_path):
    bench = request.getfixturevalue(loader)
    external.save(bench, tmp_path)
    loaded = external.load(tmp_path)
    assert dataset.content_hash(loaded) == dataset.content_hash(bench)
    assert loaded.needs_judge == bench.needs_judge
    assert loaded.writer == bench.writer and loaded.config == bench.config
    assert loaded.sleeps == bench.sleeps
    # The plain dataset loader reads the same directory; it just has no judge flags.
    plain = dataset.load(tmp_path)
    assert dataset.content_hash(plain) == dataset.content_hash(bench)

    sidecar = json.loads((tmp_path / external.SIDECAR).read_text())
    sidecar["sha256"] = "0" * 64
    (tmp_path / external.SIDECAR).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        external.load(tmp_path)


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("longmemeval_oracle.json", external.load_longmemeval),
        ("longmemeval_s_cleaned.json", external.load_longmemeval),
        ("locomo10.json", external.load_locomo),
    ],
)
def test_real_download(name, loader, tmp_path):
    """Runs only when the real file has been downloaded into data/external/ (gitignored)."""
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    bench = loader(path, limit=3)
    _valid(bench)
    external.save(bench, tmp_path)
    assert dataset.content_hash(external.load(tmp_path)) == dataset.content_hash(bench)
