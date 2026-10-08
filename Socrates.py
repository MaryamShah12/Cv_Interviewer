"""
Agent Socrates (IQ / Analytical round)
--------------------------------------
1. Loads + validates the question bank (socrates_iq_bank.json). Bad entries are skipped
   with a warning instead of crashing the interview.
2. Randomly draws 10 questions (different set AND order every attempt), balanced across
   categories and difficulty so every candidate gets a comparable test. Options are shuffled
   per question, honouring each question's `shuffle_options` flag.
3. Asks the candidate ONE question at a time using LangGraph `interrupt()`.
   The answer key never leaves the graph state while the quiz is running.
4. Scores the quiz (each question's `marks`, 1 each -> 10 total) and saves the result.

No LLM is used: the round is fully deterministic apart from the random draw.
`time_limit_sec` in the bank is intentionally ignored (no timing).

Bank format (what this file expects):
    {
      "questions": [
        {
          "id": "LOG-001",                  # required, unique
          "category": "logical",            # logical | quantitative | pattern | data_interpretation
          "difficulty": "easy",             # easy | medium | hard
          "marks": 1,                       # optional, default 1
          "question": "...",
          "options": ["...", "...", "...", "..."],   # 2-6 unique options
          "correct_index": 1,               # 0-based index into the ORIGINAL options
          "shuffle_options": true,          # optional, default true
          "explanation": "..."              # optional
        }
      ]
    }
(A bare JSON list of such objects also works.)

CLI usage:
    python socrates.py <candidate_id> [bank_path]

Programmatic usage (e.g. from a FastAPI backend):
    quiz = SocratesQuiz("candidate_123")        # bank from SOCRATES_BANK_PATH or default path
    step = quiz.start()          # {"status": "awaiting_answer", "question": ..., "options": ...}
    step = quiz.answer("b")      # next question ... or {"status": "completed", "score": 7, ...}
"""
import json
import os
import random
import re
import string
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

TOTAL_QUESTIONS = 10
DEFAULT_MARKS = 1
MIN_OPTIONS = 2
MAX_OPTIONS = 6
DEFAULT_BANK_PATH = "socrates_iq_bank.json"
RECURSION_LIMIT = 1000  # every invalid answer costs one graph step; never let that crash a quiz


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class SocratesState(TypedDict):
    candidate_id: str
    bank_path: str
    requested_total: int
    balanced: bool                 # stratify the draw by category + difficulty
    seed: int | None               # only for reproducible tests; None = truly random
    input_error: str | None
    bank_warnings: list            # bank entries skipped while loading
    # ---- quiz state ----
    quiz: list                     # the selected questions + answer key (never sent mid-quiz)
    total: int                     # number of questions in this quiz
    current_index: int
    answers: list                  # candidate's answers so far (option indexes, post-shuffle)
    last_input_error: str | None
    score: float | int | None
    total_marks: float | int
    review: list
    category_scores: dict
    difficulty_scores: dict
    result_path: str | None


# --------------------------------------------------------------------------- #
# Question bank loading + validation
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")  # also strips a BOM if present
    except UnicodeDecodeError:
        return raw.decode("cp1252")  # Windows ANSI fallback


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_item(item) -> dict:
    """Validate one bank entry. Raises ValueError with a human-readable reason."""
    if not isinstance(item, dict):
        raise ValueError("entry is not an object")

    qid = str(item.get("id") or "").strip()
    if not qid:
        raise ValueError("missing id")

    question = str(item.get("question") or "").strip()
    if len(question) < 3:
        raise ValueError(f"{qid}: missing question text")

    raw_options = item.get("options")
    if not isinstance(raw_options, list):
        raise ValueError(f"{qid}: options must be a list")
    if not all(isinstance(o, str) or _is_number(o) for o in raw_options):
        raise ValueError(f"{qid}: every option must be text or a number")
    options = [str(o).strip() for o in raw_options]
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{qid}: needs {MIN_OPTIONS}-{MAX_OPTIONS} options, found {len(options)}")
    if any(not o for o in options):
        raise ValueError(f"{qid}: contains an empty option")
    if len({o.lower() for o in options}) != len(options):
        raise ValueError(f"{qid}: contains duplicate options")

    correct = item.get("correct_index")
    if isinstance(correct, bool) or not isinstance(correct, int):
        raise ValueError(f"{qid}: correct_index must be an integer")
    if not 0 <= correct < len(options):
        raise ValueError(f"{qid}: correct_index {correct} is out of range "
                         f"(0-based, {len(options)} options)")

    marks = item.get("marks", DEFAULT_MARKS)
    if not _is_number(marks) or marks <= 0:
        raise ValueError(f"{qid}: marks must be a positive number")

    shuffle = item.get("shuffle_options", True)
    if not isinstance(shuffle, bool):
        raise ValueError(f"{qid}: shuffle_options must be true or false")

    return {
        "id": qid,
        "category": str(item.get("category") or "general").strip().lower() or "general",
        "difficulty": str(item.get("difficulty") or "unspecified").strip().lower() or "unspecified",
        "marks": marks,
        "question": question,
        "options": options,
        "correct_index": correct,
        "shuffle_options": shuffle,
        "explanation": str(item.get("explanation") or "").strip(),
    }


def load_bank(path: Path) -> tuple[list[dict], list[str]]:
    """Return (valid questions, warnings). Raises ValueError only if the file is unusable."""
    if not path.exists():
        raise ValueError(f"Question bank not found: {path}")
    try:
        data = json.loads(_read_text(path))
    except ValueError as e:  # JSONDecodeError is a ValueError
        raise ValueError(f"Question bank is not valid JSON: {e}")

    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list):
        raise ValueError("Question bank must contain a 'questions' list")

    bank, warnings, seen_ids, seen_text = [], [], set(), set()
    for pos, item in enumerate(data, start=1):
        try:
            q = _parse_item(item)
            if q["id"] in seen_ids:
                raise ValueError(f"{q['id']}: duplicate id")
            text_key = re.sub(r"\s+", " ", q["question"].lower())
            if text_key in seen_text:
                raise ValueError(f"{q['id']}: duplicate question text")
        except ValueError as e:
            warnings.append(f"entry #{pos} skipped: {e}")
            continue
        seen_ids.add(q["id"])
        seen_text.add(text_key)
        bank.append(q)
    return bank, warnings


# --------------------------------------------------------------------------- #
# Random selection + option shuffling
# --------------------------------------------------------------------------- #
def _quotas(sizes: dict, total: int, rng: random.Random) -> dict:
    """Split `total` across groups in proportion to group size (largest-remainder method).

    Integer maths only. Ties are broken randomly. A group never gets more than its size.
    """
    pool = sum(sizes.values())
    quota = {k: total * n // pool for k, n in sizes.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(total * sizes[k] % pool), rng.random()))
    for k in by_remainder[: total - sum(quota.values())]:
        quota[k] += 1
    return quota


def _draw_balanced(bank: list[dict], total: int, rng: random.Random) -> list[dict]:
    """Random draw whose difficulty mix and category mix follow the bank's proportions.

    For the 60-question bank and total=10 that is exactly 4 easy / 4 medium / 2 hard and
    3 logical / 3 quantitative / 2 pattern / 2 data_interpretation (when availability allows).
    """
    pools = defaultdict(lambda: defaultdict(list))  # difficulty -> category -> questions
    for q in bank:
        pools[q["difficulty"]][q["category"]].append(q)

    diff_quota = _quotas({d: sum(map(len, c.values())) for d, c in pools.items()}, total, rng)
    cat_target = _quotas(dict(Counter(q["category"] for q in bank)), total, rng)

    slots = [d for d, n in diff_quota.items() for _ in range(n)]
    rng.shuffle(slots)

    picked, chosen = Counter(), []
    for d in slots:
        # Pick the category that is furthest below its target (random tie-break) ...
        cats = [c for c, qs in pools[d].items() if qs]
        c = min(cats, key=lambda c: (picked[c] - cat_target[c], rng.random()))
        # ... then a random unused question from that category/difficulty pool.
        pool = pools[d][c]
        chosen.append(pool.pop(rng.randrange(len(pool))))
        picked[c] += 1
    rng.shuffle(chosen)
    return chosen


def select_quiz(bank: list[dict], total: int, rng: random.Random,
                balanced: bool = True) -> list[dict]:
    """Pick `total` distinct questions in random order and shuffle their options."""
    drawn = _draw_balanced(bank, total, rng) if balanced else rng.sample(bank, total)

    quiz = []
    for q in drawn:
        order = list(range(len(q["options"])))
        if q["shuffle_options"]:
            rng.shuffle(order)
        quiz.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "marks": q["marks"],
            "question": q["question"],
            "options": [q["options"][i] for i in order],
            "correct_index": order.index(q["correct_index"]),  # follow the correct option
            "explanation": q["explanation"],
        })
    return quiz


# --------------------------------------------------------------------------- #
# Node 1: load bank + pick the quiz
# --------------------------------------------------------------------------- #
def node_prepare_quiz(state: SocratesState) -> dict:
    def fail(msg: str, warnings=()) -> dict:
        return {"input_error": msg, "bank_warnings": list(warnings), "quiz": [], "total": 0}

    if not (state.get("candidate_id") or "").strip():
        return fail("No candidate id provided.")
    total = state.get("requested_total", TOTAL_QUESTIONS)
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        return fail(f"Invalid number of questions: {total!r}")

    try:
        bank, warnings = load_bank(Path(state["bank_path"]))
    except ValueError as e:
        return fail(str(e))
    if len(bank) < total:
        return fail(f"Question bank has only {len(bank)} valid questions; need at least {total}.",
                    warnings)

    rng = random.Random(state.get("seed"))
    quiz = select_quiz(bank, total, rng, balanced=state.get("balanced", True))
    return {"quiz": quiz, "total": len(quiz), "bank_warnings": warnings, "input_error": None,
            "total_marks": sum(q["marks"] for q in quiz),
            "current_index": 0, "answers": [], "last_input_error": None}


# --------------------------------------------------------------------------- #
# Node 2: ask ONE question (human-in-the-loop via interrupt)
# --------------------------------------------------------------------------- #
def normalize_answer(raw, n_options: int) -> int | None:
    """Accept 'a', 'B', 'c)', '(d)', 'option b', or '1'..'n'. Return option index or None."""
    s = str(raw).strip().lower()
    m = re.fullmatch(r"(?:option\s*)?\(?([a-z]|\d)\)?[.):]?", s)
    if not m:
        return None
    ch = m.group(1)
    idx = int(ch) - 1 if ch.isdigit() else string.ascii_lowercase.index(ch)
    return idx if 0 <= idx < n_options else None


def node_ask_question(state: SocratesState) -> dict:
    from langgraph.types import interrupt

    quiz = state["quiz"]
    idx = state["current_index"]
    q = quiz[idx]
    letters = string.ascii_lowercase[:len(q["options"])]

    # NOTE: only question + options go out. Answer key, difficulty and explanation stay
    # in graph state.
    payload = {
        "question_number": idx + 1,
        "total": len(quiz),
        "category": q["category"],
        "marks": q["marks"],
        "question": q["question"],
        "options": dict(zip(letters, q["options"])),
        "error": state.get("last_input_error"),
    }

    # Graph pauses here. The value passed to Command(resume=...) is returned.
    # (On resume this node re-runs from the top, so nothing above may have side effects.)
    raw = interrupt(payload)

    choice = normalize_answer(raw, len(q["options"]))
    if choice is None:
        # invalid input: stay on the same question and re-ask
        return {"last_input_error": f"Invalid answer {str(raw)!r}. "
                                    f"Please reply with one of: {', '.join(letters)}."}
    return {"answers": list(state["answers"]) + [choice],
            "current_index": idx + 1,
            "last_input_error": None}


# --------------------------------------------------------------------------- #
# Node 3: score (each question's marks)
# --------------------------------------------------------------------------- #
def _tally(bucket: dict, key: str, ok: bool, marks) -> None:
    row = bucket.setdefault(key, {"correct": 0, "total": 0, "marks": 0, "max_marks": 0})
    row["total"] += 1
    row["max_marks"] += marks
    if ok:
        row["correct"] += 1
        row["marks"] += marks


def node_score(state: SocratesState) -> dict:
    review, by_category, by_difficulty, score = [], {}, {}, 0
    for i, (q, ans) in enumerate(zip(state["quiz"], state["answers"]), start=1):
        letters = string.ascii_lowercase[:len(q["options"])]
        ok = ans == q["correct_index"]
        earned = q["marks"] if ok else 0
        score += earned
        _tally(by_category, q["category"], ok, q["marks"])
        _tally(by_difficulty, q["difficulty"], ok, q["marks"])

        review.append({
            "question_number": i,
            "question_id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "options": dict(zip(letters, q["options"])),
            "your_answer": letters[ans],
            "your_answer_text": q["options"][ans],
            "correct_option": letters[q["correct_index"]],
            "correct_answer_text": q["options"][q["correct_index"]],
            "is_correct": ok,
            "marks": earned,
            "max_marks": q["marks"],
            "explanation": q["explanation"],
        })
    return {"score": score, "review": review,
            "category_scores": by_category, "difficulty_scores": by_difficulty}


# --------------------------------------------------------------------------- #
# Node 4: save result
# --------------------------------------------------------------------------- #
def _safe_stem(candidate_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", candidate_id.strip()).strip("._")
    return stem or "candidate"


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # a reader never sees a half-written file


def node_save(state: SocratesState) -> dict:
    out_dir = Path(os.getenv("SOCRATES_OUTPUT_DIR", "socrates_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_stem(state.get('candidate_id') or '')}_socrates_result.json"

    payload = {
        "candidate_id": state.get("candidate_id"),
        "round": "iq_analytical",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank_warnings": state.get("bank_warnings", []),
    }
    if state.get("score") is not None:
        payload.update({
            "status": "completed",
            "score": state["score"],
            "total_marks": state["total_marks"],
            "category_scores": state["category_scores"],
            "difficulty_scores": state["difficulty_scores"],
            "review": state["review"],
        })
    else:
        payload.update({"status": "error", "error": state.get("input_error") or "Unknown error"})

    _write_json_atomic(out_path, payload)
    return {"result_path": str(out_path)}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(SocratesState)
    g.add_node("prepare_quiz", node_prepare_quiz)
    g.add_node("ask_question", node_ask_question)
    g.add_node("score", node_score)
    g.add_node("save", node_save)

    def route_after_prepare(state):
        return "save" if state.get("input_error") else "ask_question"

    def route_after_question(state):
        return "ask_question" if state["current_index"] < state["total"] else "score"

    g.add_edge(START, "prepare_quiz")
    g.add_conditional_edges("prepare_quiz", route_after_prepare,
                            {"save": "save", "ask_question": "ask_question"})
    g.add_conditional_edges("ask_question", route_after_question,
                            {"ask_question": "ask_question", "score": "score"})
    g.add_edge("score", "save")
    g.add_edge("save", END)

    # interrupt() REQUIRES a checkpointer. Swap MemorySaver for SqliteSaver /
    # PostgresSaver if the quiz must survive a server restart (and for a stateless backend).
    return g.compile(checkpointer=checkpointer or MemorySaver())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class SocratesQuiz:
    """Drives the graph: start() -> answer() -> answer() ... -> completed."""

    def __init__(self, candidate_id: str, bank_path: str | None = None,
                 total_questions: int = TOTAL_QUESTIONS, balanced: bool = True,
                 seed: int | None = None, checkpointer=None, thread_id: str | None = None,
                 reveal_review: bool = False):
        self.graph = build_graph(checkpointer)
        self.reveal_review = reveal_review  # False: candidate sees score only, not the key
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
                       "recursion_limit": RECURSION_LIMIT}
        self._init: SocratesState = {
            "candidate_id": candidate_id,
            "bank_path": bank_path or os.getenv("SOCRATES_BANK_PATH", DEFAULT_BANK_PATH),
            "requested_total": total_questions, "balanced": balanced, "seed": seed,
            "input_error": None, "bank_warnings": [],
            "quiz": [], "total": 0, "current_index": 0, "answers": [],
            "last_input_error": None, "score": None, "total_marks": 0, "review": [],
            "category_scores": {}, "difficulty_scores": {}, "result_path": None,
        }

    def start(self) -> dict:
        # Resuming an existing thread (same thread_id + persistent checkpointer)?
        # Never restart it - that would re-draw the questions.
        if self.graph.get_state(self.config).values:
            return self._status()
        self.graph.invoke(self._init, self.config)
        return self._status()

    def answer(self, choice: str) -> dict:
        from langgraph.types import Command

        status = self._status()
        if status["status"] != "awaiting_answer":  # nothing to answer (finished / not started)
            return status
        self.graph.invoke(Command(resume=choice), self.config)
        return self._status()

    def _status(self) -> dict:
        snap = self.graph.get_state(self.config)
        if not snap.values:
            return {"status": "error", "error": "Quiz has not been started."}
        if snap.next:  # paused at an interrupt
            for task in snap.tasks:
                if task.interrupts:
                    return {"status": "awaiting_answer", **task.interrupts[0].value}
            return {"status": "error", "error": "Quiz is in an unexpected state."}

        v = snap.values
        if v.get("score") is not None:
            out = {"status": "completed", "score": v["score"], "total_marks": v["total_marks"],
                   "percentage": round(100 * v["score"] / v["total_marks"], 1),
                   "category_scores": v["category_scores"],
                   "difficulty_scores": v["difficulty_scores"],
                   "result_path": v.get("result_path")}
            if self.reveal_review:
                out["review"] = v["review"]
            return out
        return {"status": "error", "error": v.get("input_error") or "Unknown error",
                "result_path": v.get("result_path")}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pretty(category: str) -> str:
    return category.replace("_", " ").title()


def run_cli(candidate_id: str, bank_path: str | None = None) -> None:
    quiz = SocratesQuiz(candidate_id, bank_path, reveal_review=True)
    step = quiz.start()

    while step["status"] == "awaiting_answer":
        if step.get("error"):
            print(f"\n  ! {step['error']}")
        print(f"\nQuestion {step['question_number']}/{step['total']}  "
              f"[{_pretty(step['category'])}]  ({step['marks']} mark)")
        print(step["question"])
        for k, text in step["options"].items():
            print(f"  {k.upper()}) {text}")
        letters = "/".join(step["options"])
        step = quiz.answer(input(f"Your answer ({letters}): "))

    if step["status"] == "error":
        print("\nCould not run the quiz:", step["error"])
        return

    print(f"\n=== RESULT: {step['score']} / {step['total_marks']} ({step['percentage']}%) ===")
    for cat, s in step["category_scores"].items():
        print(f"  {_pretty(cat)}: {s['correct']}/{s['total']}")
    for r in step["review"]:
        mark = "✔" if r["is_correct"] else "✘"
        print(f"\nQ{r['question_number']} {mark}  your answer: {r['your_answer'].upper()}"
              f"  correct: {r['correct_option'].upper()}")
        if not r["is_correct"] and r["explanation"]:
            print(f"   Why: {r['explanation']}")
    print(f"\nSaved result to: {step['result_path']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python socrates.py <candidate_id> [bank_path]")
    run_cli(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)