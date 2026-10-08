"""
Agent Lovelace (Coding round) - simple version
----------------------------------------------
1. Loads the candidate CV JSON (skills / projects) and the role.
2. Asks the LLM for ONE coding problem grounded in the role + CV.
   (Just a problem: title, statement, examples, constraints. No hidden tests,
   no reference solution.)
3. Shows the problem to the candidate (LangGraph `interrupt()`).
4. The candidate writes code in a textbox (any of Python / JavaScript / Java / C++).
   Submissions are only SAVED (no LLM call), so resubmitting is free.
5. On finalize, ONE LLM call reads the latest code against the problem and
   awards 20 marks: correctness 12 + approach 4 + clarity 4.
   No code is executed; trivial syntax slips are forgiven.

Total LLM calls per round: 1 (problem) + 1 (grading).

CLI:
    python lovelace.py <cv_json_path> "<role>" [--candidate-id ID]

Programmatic:
    round = LovelaceRound(cv_json_path, role, candidate_id="c123")
    step = round.start()                 # {"status": "awaiting_submission", "problem": {...}}
    step = round.submit(code, "python")  # {"status": "awaiting_submission", "message": "saved"}
    step = round.finalize()              # {"status": "completed", "score": 15.0, ...}

Environment (.env works): GEMINI_API_KEY (required), LOVELACE_MODEL
(default: gemini-2.5-flash), LOVELACE_OUTPUT_DIR (default: lovelace_results).
"""
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TOTAL_MARKS = 20
CORRECTNESS_MARKS = 12   # does the code solve the problem?  (reported as test_marks)
QUALITY_MARKS = 8        # approach 4 + clarity 4            (reported as review_marks)
MAX_GENERATION_ATTEMPTS = 3
MAX_GRADE_ATTEMPTS = 3
MAX_CODE_CHARS = 30_000
RECURSION_LIMIT = 500

SUPPORTED_LANGUAGES = ("python", "javascript", "java", "cpp")
LANGUAGE_ALIASES = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "javascript", "ts": "javascript",
    "java": "java", "c++": "cpp", "cpp": "cpp", "cplusplus": "cpp",
}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Example(BaseModel):
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    explanation: str = ""


class CodingProblem(BaseModel):
    title: str = Field(min_length=3)
    statement: str = Field(min_length=50)
    examples: list[Example] = Field(min_length=1, max_length=3)
    constraints: list[str] = Field(default_factory=list)
    concepts_tested: list[str] = Field(default_factory=list)
    time_estimate_min: int = Field(ge=5, le=30, default=15)


class GradeResult(BaseModel):
    correctness: float = Field(ge=0, le=CORRECTNESS_MARKS)
    approach: float = Field(ge=0, le=4)
    clarity: float = Field(ge=0, le=4)
    feedback: str = Field(min_length=1)
    strengths: list[str] = []
    improvements: list[str] = []


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class LovelaceState(TypedDict):
    cv_json_path: str
    role: str
    candidate_id: str
    skills: list
    projects: list
    cv_languages: list
    input_error: str | None
    problem: dict | None
    generation_error: str | None
    attempts: int
    pending_action: str | None      # saved | finalize | reask
    last_input_error: str | None
    last_message: str | None
    submissions: list
    review: dict | None
    review_error: str | None
    score: float | None
    test_marks: float | None
    review_marks: float | None
    presented_at: str | None
    finalized_at: str | None
    problem_path: str | None
    result_path: str | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_language(raw) -> str | None:
    return LANGUAGE_ALIASES.get(str(raw or "").strip().lower())


def _read_text_fallback(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value
            if isinstance(x, (str, int, float)) and str(x).strip()]


def _detect_languages(skills: list[str]) -> list[str]:
    found = []
    for s in skills:
        lang = LANGUAGE_ALIASES.get(s.strip().lower())
        if lang and lang not in found:
            found.append(lang)
    return found


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def extract_json(raw: str):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(cleaned[start:end + 1])


def _llm_json(llm, messages) -> dict:
    """Call the LLM and return a parsed JSON object, with a useful error if it fails."""
    response = llm.invoke(messages)
    raw = _content_to_text(response.content if hasattr(response, "content") else response)
    if not raw.strip():
        meta = getattr(response, "response_metadata", {}) or {}
        raise ValueError(f"LLM returned EMPTY text (finish_reason={meta.get('finish_reason')})")
    try:
        data = extract_json(raw)
    except ValueError:
        raise ValueError(f"No JSON in LLM output. Started with: {raw[:150]!r}")
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip()).strip("._")
    return stem or "candidate"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
GENERATION_SYSTEM = """You are Agent Lovelace, a senior engineer who designs coding interview questions.

Design ONE coding problem for a candidate, tailored to the role they applied for and
inspired by the skills and projects on their CV.

Rules:
1. The problem must test understanding of a concept (data structure, algorithm, edge
   cases), not memorised syntax.
2. Solvable in 10-15 minutes, about 10-30 lines of code. A single function or small
   program. It must be FULLY self-contained: solvable without ever seeing the CV.
3. Language-agnostic: do not tie the problem to one programming language.
4. Describe clearly what the input is and what the output must be.
5. Give 1-3 examples (input, output, short explanation) and a few constraints.
6. No randomness, dates/times, files, or network access.
7. Output ONE valid JSON object matching the schema. No markdown, no code fences,
   no text before or after the JSON.

=== INPUT ===
Role: {role}
Candidate skills: {skills}
Candidate projects: {projects}

=== OUTPUT SCHEMA ===
{schema}"""

GRADING_SYSTEM = """You are Agent Lovelace's code judge, a senior engineer grading a coding
interview by READING the candidate's code, like a human interviewer. Nothing is executed.

You get the problem and the candidate's code (written in {language}). Award marks:

- correctness (0-{cmax}): does the logic actually solve the problem as stated, including
  the examples and sensible edge cases (empty input, single element, duplicates,
  boundaries)? Trace the examples through the code. Wrong logic, a missed requirement,
  or code that cannot produce the right output loses marks. Partial credit is fine for
  a mostly-correct solution with a specific flaw.
- approach (0-4): right algorithm / data structure and reasonable time/space complexity.
- clarity (0-4): readable, sensibly named, well structured.

Rules:
- FORGIVE trivial syntax slips (a missing colon, semicolon or bracket) when the intent
  is clear. Judge what the candidate MEANT. Be STRICT about logic.
- Empty, unrelated, or placeholder code (e.g. only `pass`) scores 0 everywhere.
- Be concise and specific; point to the exact lines or cases that are wrong.
- Output ONE valid JSON object matching the schema. No markdown, no code fences,
  no text before or after the JSON.

=== OUTPUT SCHEMA ===
{schema}"""


def build_generation_prompt(skills, projects, role, repair_hint=None) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    sys_msg = SystemMessage(content=GENERATION_SYSTEM.format(
        role=role, skills=", ".join(skills),
        projects=json.dumps(projects, ensure_ascii=False),
        schema=json.dumps(CodingProblem.model_json_schema(), indent=2)))
    user = "Design the coding problem now."
    if repair_hint:
        user = (f"Your previous output failed with this error:\n{repair_hint}\n"
                f"Return the full corrected JSON.\n\n" + user)
    return [sys_msg, HumanMessage(content=user)]


def build_grading_prompt(problem: dict, code: str, language: str) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    sys_msg = SystemMessage(content=GRADING_SYSTEM.format(
        language=language, cmax=CORRECTNESS_MARKS,
        schema=json.dumps(GradeResult.model_json_schema(), indent=2)))
    shown = {k: problem.get(k) for k in ("title", "statement", "examples", "constraints")}
    user = (f"Problem:\n{json.dumps(shown, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate's {language} code:\n```\n{code}\n```\n\nGrade it now.")
    return [sys_msg, HumanMessage(content=user)]


# --------------------------------------------------------------------------- #
# Node 1: load + validate CV
# --------------------------------------------------------------------------- #
def node_load_validate(state: LovelaceState) -> dict:
    path = Path(state["cv_json_path"])
    role = (state.get("role") or "").strip()

    try:
        if not path.exists():
            raise ValueError("CV JSON file does not exist")
        cv = json.loads(_read_text_fallback(path))
        if not isinstance(cv, dict):
            raise ValueError("CV JSON root must be an object")
    except ValueError as e:
        return {"skills": [], "projects": [], "cv_languages": [],
                "input_error": f"Cannot load CV JSON: {e}"}

    skills = cv.get("skills")
    skills = skills if isinstance(skills, dict) else {}
    merged = list(dict.fromkeys(
        _as_str_list(skills.get("technical")) + _as_str_list(skills.get("tools"))))

    projects = []
    raw_projects = cv.get("projects") if isinstance(cv.get("projects"), list) else []
    for p in raw_projects:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        if name:
            projects.append({"name": name,
                             "description": str(p.get("description") or "").strip(),
                             "technologies": _as_str_list(p.get("technologies"))})

    if not role:
        return {"skills": [], "projects": [], "cv_languages": [],
                "input_error": "No role selected."}
    if len(merged) < 2:
        return {"skills": merged, "projects": [], "cv_languages": [],
                "input_error": "CV has fewer than 2 extractable skills."}
    if not projects:
        return {"skills": merged, "projects": [], "cv_languages": [],
                "input_error": "CV has no extractable projects."}
    return {"skills": merged, "projects": projects,
            "cv_languages": _detect_languages(merged), "input_error": None}


# --------------------------------------------------------------------------- #
# Node 2: generate the problem (1 LLM call per attempt)
# --------------------------------------------------------------------------- #
def node_generate_problem(state: LovelaceState, llm) -> dict:
    attempts = state.get("attempts", 0) + 1
    repair = state.get("generation_error") if attempts > 1 else None
    try:
        data = _llm_json(llm, build_generation_prompt(
            state["skills"], state["projects"], state["role"], repair_hint=repair))
        return {"problem": data, "attempts": attempts, "generation_error": None}
    except Exception as e:  # noqa: BLE001
        if attempts < MAX_GENERATION_ATTEMPTS:
            time.sleep(2 * attempts)  # gentle backoff before the retry
        return {"problem": None, "attempts": attempts,
                "generation_error": f"generation attempt {attempts} failed: {e}"}


# --------------------------------------------------------------------------- #
# Node 3: validate the problem (schema only)
# --------------------------------------------------------------------------- #
def node_validate_problem(state: LovelaceState) -> dict:
    if state.get("problem") is None:
        return {}  # generation failed; generation_error already set
    try:
        data = CodingProblem.model_validate(state["problem"]).model_dump()
    except ValidationError as e:
        return {"problem": None, "generation_error": str(e)[:600]}
    data["role"] = state["role"]
    return {"problem": data, "generation_error": None}


# --------------------------------------------------------------------------- #
# Node 4: save the problem
# --------------------------------------------------------------------------- #
def node_save_problem(state: LovelaceState) -> dict:
    out_dir = Path(os.getenv("LOVELACE_OUTPUT_DIR", "lovelace_results"))
    stem = _safe_stem(state.get("candidate_id") or Path(state["cv_json_path"]).stem)
    out_path = out_dir / f"{stem}_lovelace_problem.json"

    payload: dict = {"candidate_id": state.get("candidate_id"), "role": state.get("role"),
                     "generated_at": datetime.now(timezone.utc).isoformat()}
    if state.get("problem"):
        payload.update({"status": "ready", "problem": state["problem"]})
    else:
        payload.update({"status": "error",
                        "error": state.get("input_error")
                        or state.get("generation_error") or "Unknown error"})
    _write_json_atomic(out_path, payload)
    return {"problem_path": str(out_path),
            "presented_at": datetime.now(timezone.utc).isoformat()
            if state.get("problem") else None}


def public_problem(problem: dict) -> dict:
    return {
        "title": problem["title"],
        "statement": problem["statement"],
        "examples": problem["examples"],
        "constraints": problem["constraints"],
        "concepts_tested": problem["concepts_tested"],
        "time_estimate_min": problem["time_estimate_min"],
        "total_marks": TOTAL_MARKS,
        "marking_scheme": (f"{CORRECTNESS_MARKS} marks: correctness of your logic. "
                           f"4 marks: approach / efficiency. 4 marks: code clarity."),
    }


# --------------------------------------------------------------------------- #
# Node 5: await a submission (human-in-the-loop). Submissions are only SAVED.
# --------------------------------------------------------------------------- #
def node_await_submission(state: LovelaceState) -> dict:
    from langgraph.types import interrupt

    subs = list(state.get("submissions") or [])
    payload = {
        "problem": public_problem(state["problem"]),
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "detected_languages": state.get("cv_languages") or [],
        "attempts_used": len(subs),
        "message": state.get("last_message"),
        "error": state.get("last_input_error"),
        # Resume with {"action": "submit", "code": "...", "language": "python"}
        # or {"action": "finalize"}.
    }
    resume = interrupt(payload)   # graph pauses here

    if not isinstance(resume, dict):
        return {"last_input_error": 'Malformed response. Use {"action": "submit", ...} '
                                    'or {"action": "finalize"}.',
                "last_message": None, "pending_action": "reask"}

    action = resume.get("action")
    if action == "finalize":
        return {"pending_action": "finalize", "last_input_error": None, "last_message": None}

    if action == "submit":
        code = resume.get("code")
        language = normalize_language(resume.get("language"))
        if not isinstance(code, str) or not code.strip():
            return {"last_input_error": "Empty submission. Paste your code and submit again.",
                    "last_message": None, "pending_action": "reask"}
        if len(code) > MAX_CODE_CHARS:
            return {"last_input_error": f"Code is too long ({len(code)} chars, max {MAX_CODE_CHARS}).",
                    "last_message": None, "pending_action": "reask"}
        if language is None:
            return {"last_input_error": f"Unsupported language {resume.get('language')!r}. "
                                        f"Choose one of: {', '.join(SUPPORTED_LANGUAGES)}.",
                    "last_message": None, "pending_action": "reask"}
        subs.append({"n": len(subs) + 1, "language": language, "code": code,
                     "at": datetime.now(timezone.utc).isoformat()})
        return {"submissions": subs, "pending_action": "saved", "last_input_error": None,
                "last_message": f"Submission #{len(subs)} saved. Submit again to replace it, "
                                f"or finish to get graded."}

    return {"last_input_error": f"Unknown action {action!r}. Use 'submit' or 'finalize'.",
            "last_message": None, "pending_action": "reask"}


# --------------------------------------------------------------------------- #
# Node 6: grade the latest submission (1 LLM call)
# --------------------------------------------------------------------------- #
def node_grade(state: LovelaceState, llm) -> dict:
    subs = state.get("submissions") or []
    if not subs:
        # Nothing submitted: zero marks, no LLM call.
        return {"review": {"correctness": 0, "approach": 0, "clarity": 0, "total": 0,
                           "feedback": "No code was submitted.",
                           "strengths": [], "improvements": []},
                "review_error": "Candidate finalized without submitting any code."}

    latest = subs[-1]
    last_err = ""
    for attempt in range(1, MAX_GRADE_ATTEMPTS + 1):
        try:
            data = _llm_json(llm, build_grading_prompt(
                state["problem"], latest["code"], latest["language"]))
            g = GradeResult.model_validate(data).model_dump()
            g["total"] = round(g["correctness"] + g["approach"] + g["clarity"], 1)
            return {"review": g, "review_error": None}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:200]
            if attempt < MAX_GRADE_ATTEMPTS:
                time.sleep(2 * attempt)

    # Grading failed: let the candidate retry finalize instead of losing the round.
    return {"review": None,
            "review_error": f"Grading is temporarily unavailable ({last_err}). "
                            f"Press finish again to retry."}


# --------------------------------------------------------------------------- #
# Node 7: score
# --------------------------------------------------------------------------- #
def node_score(state: LovelaceState) -> dict:
    r = state.get("review") or {}
    test_marks = round(float(r.get("correctness", 0)), 1)
    review_marks = round(float(r.get("approach", 0)) + float(r.get("clarity", 0)), 1)
    score = round(min(TOTAL_MARKS, test_marks + review_marks), 1)
    return {"test_marks": test_marks, "review_marks": review_marks, "score": score,
            "finalized_at": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------- #
# Node 8: save the result
# --------------------------------------------------------------------------- #
def node_save_result(state: LovelaceState) -> dict:
    out_dir = Path(os.getenv("LOVELACE_OUTPUT_DIR", "lovelace_results"))
    stem = _safe_stem(state.get("candidate_id") or Path(state["cv_json_path"]).stem)
    out_path = out_dir / f"{stem}_lovelace_result.json"
    subs = state.get("submissions") or []
    payload = {
        "candidate_id": state.get("candidate_id"),
        "role": state.get("role"),
        "round": "coding",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "presented_at": state.get("presented_at"),
        "finalized_at": state.get("finalized_at"),
        "status": "completed" if state.get("score") is not None else "error",
        "score": state.get("score"),
        "total_marks": TOTAL_MARKS,
        "test_marks": state.get("test_marks"),
        "max_test_marks": CORRECTNESS_MARKS,
        "review_marks": state.get("review_marks"),
        "max_review_marks": QUALITY_MARKS,
        "review": state.get("review"),
        "review_error": state.get("review_error"),
        "attempts_used": len(subs),
        "final_submission": subs[-1] if subs else None,
        "problem": state.get("problem"),
        "problem_path": state.get("problem_path"),
    }
    _write_json_atomic(out_path, payload)
    return {"result_path": str(out_path)}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(llm, checkpointer=None, grader_llm=None):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(LovelaceState)
    g.add_node("load_validate", node_load_validate)
    g.add_node("generate_problem", lambda s: node_generate_problem(s, llm))
    g.add_node("validate_problem", node_validate_problem)
    g.add_node("save_problem", node_save_problem)
    g.add_node("await_submission", node_await_submission)
    g.add_node("grade", lambda s: node_grade(s, grader_llm or llm))
    g.add_node("score", node_score)
    g.add_node("save_result", node_save_result)

    def after_input_check(state):
        return "save_problem" if state.get("input_error") else "generate_problem"

    def after_validation(state):
        if state.get("problem") and not state.get("generation_error"):
            return "save_problem"
        if state.get("attempts", 0) < MAX_GENERATION_ATTEMPTS:
            return "generate_problem"
        return "save_problem"          # give up; the error is recorded

    def after_problem_saved(state):
        return "await_submission" if state.get("problem") else END

    def after_submission(state):
        return "grade" if state.get("pending_action") == "finalize" else "await_submission"

    def after_grade(state):
        # Grader failed -> go back to the candidate so they can retry finalize.
        return "score" if state.get("review") else "await_submission"

    g.add_edge(START, "load_validate")
    g.add_conditional_edges("load_validate", after_input_check,
                            {"save_problem": "save_problem", "generate_problem": "generate_problem"})
    g.add_edge("generate_problem", "validate_problem")
    g.add_conditional_edges("validate_problem", after_validation,
                            {"save_problem": "save_problem", "generate_problem": "generate_problem"})
    g.add_conditional_edges("save_problem", after_problem_saved,
                            {"await_submission": "await_submission", END: END})
    g.add_conditional_edges("await_submission", after_submission,
                            {"grade": "grade", "await_submission": "await_submission"})
    g.add_conditional_edges("grade", after_grade,
                            {"score": "score", "await_submission": "await_submission"})
    g.add_edge("score", "save_result")
    g.add_edge("save_result", END)

    # interrupt() REQUIRES a checkpointer (use SqliteSaver/PostgresSaver to survive restarts).
    return g.compile(checkpointer=checkpointer or MemorySaver())


def get_llm(temperature: float = 0.4):
    """Gemini (free tier). API key from GEMINI_API_KEY."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = dict(
        model=os.getenv("LOVELACE_MODEL", "gemini-2.5-flash"),
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=temperature,
        max_output_tokens=8000,
        max_retries=6,                        # built-in backoff on 429s
        response_mime_type="application/json",
    )
    try:
        # Stops hidden "thinking" tokens from eating the output budget (empty replies).
        return ChatGoogleGenerativeAI(thinking_budget=0, **kwargs)
    except Exception:  # older library or a model that can't disable thinking
        return ChatGoogleGenerativeAI(**kwargs)


def get_grader_llm(base_llm=None):
    """Same model at a low temperature for consistent grading."""
    if base_llm is None:
        return get_llm(0.1)
    try:
        return base_llm.model_copy(update={"temperature": 0.1})
    except Exception:
        return base_llm


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class LovelaceRound:
    """Drives the graph: start() -> submit() ... -> finalize() -> completed."""

    def __init__(self, cv_json_path: str, role: str, candidate_id: str | None = None,
                 llm=None, checkpointer=None, thread_id: str | None = None):
        base = llm or get_llm()
        self.graph = build_graph(base, checkpointer, grader_llm=get_grader_llm(base))
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
                       "recursion_limit": RECURSION_LIMIT}
        self._init: LovelaceState = {
            "cv_json_path": cv_json_path, "role": role,
            "candidate_id": (candidate_id or "").strip(),
            "skills": [], "projects": [], "cv_languages": [],
            "input_error": None, "problem": None, "generation_error": None,
            "attempts": 0, "pending_action": None, "last_input_error": None,
            "last_message": None, "submissions": [], "review": None,
            "review_error": None, "score": None, "test_marks": None,
            "review_marks": None, "presented_at": None, "finalized_at": None,
            "problem_path": None, "result_path": None,
        }

    def start(self) -> dict:
        if self.graph.get_state(self.config).values:   # resuming: never regenerate
            return self._status()
        self.graph.invoke(self._init, self.config)
        return self._status()

    def submit(self, code: str, language: str = "python") -> dict:
        """Save the code (no grading yet). Submitting again replaces it."""
        from langgraph.types import Command

        if self._status()["status"] != "awaiting_submission":
            return self._status()
        self.graph.invoke(
            Command(resume={"action": "submit", "code": code, "language": language}),
            self.config)
        return self._status()

    def finalize(self) -> dict:
        """Grade the latest submission and end the round."""
        from langgraph.types import Command

        if self._status()["status"] != "awaiting_submission":
            return self._status()
        self.graph.invoke(Command(resume={"action": "finalize"}), self.config)
        status = self._status()
        if status["status"] == "awaiting_submission":
            # Grader failed; surface the reason so the caller can show it and retry.
            snap = self.graph.get_state(self.config).values
            status["error"] = snap.get("review_error") or status.get("error")
        return status

    def _status(self) -> dict:
        snap = self.graph.get_state(self.config)
        if not snap.values:
            return {"status": "error", "error": "Round has not been started."}
        if snap.next:  # paused at the interrupt
            for task in snap.tasks:
                if task.interrupts:
                    return {"status": "awaiting_submission", **task.interrupts[0].value}
            return {"status": "error", "error": "Round is in an unexpected state."}

        v = snap.values
        if v.get("score") is not None:
            subs = v.get("submissions") or []
            return {"status": "completed",
                    "score": v["score"], "total_marks": TOTAL_MARKS,
                    "percentage": round(100 * v["score"] / TOTAL_MARKS, 1),
                    "test_marks": v.get("test_marks"), "review_marks": v.get("review_marks"),
                    "review": v.get("review"), "review_error": v.get("review_error"),
                    "attempts_used": len(subs),
                    "problem_path": v.get("problem_path"),
                    "result_path": v.get("result_path")}
        return {"status": "error",
                "error": v.get("input_error") or v.get("generation_error") or "Unknown error",
                "problem_path": v.get("problem_path")}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_code() -> str:
    print("Paste your code below, then type END on its own line and press Enter.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def print_problem(p: dict) -> None:
    print(f"\n{'=' * 62}\n{p['title']}  ({p['total_marks']} marks)\n{'=' * 62}")
    print(p["statement"])
    print("\nExamples:")
    for i, e in enumerate(p["examples"], 1):
        print(f"  {i}. input:  {e['input']}")
        print(f"     output: {e['output']}")
        if e.get("explanation"):
            print(f"     why:    {e['explanation']}")
    if p.get("constraints"):
        print("\nConstraints:")
        for c in p["constraints"]:
            print(f"  - {c}")
    print(f"\nSuggested time: ~{p['time_estimate_min']} minutes")
    print(f"Marking: {p['marking_scheme']}")


def print_result(step: dict) -> None:
    print(f"\n=== RESULT: {step['score']} / {step['total_marks']} ({step['percentage']}%) ===")
    print(f"  Correctness:         {step['test_marks']} / {CORRECTNESS_MARKS}")
    print(f"  Approach + clarity:  {step['review_marks']} / {QUALITY_MARKS}")
    rv = step.get("review") or {}
    if rv.get("feedback"):
        print(f"\nFeedback:\n  {rv['feedback']}")
    for s in rv.get("strengths", []):
        print(f"  + {s}")
    for imp in rv.get("improvements", []):
        print(f"  - {imp}")


def run_cli(cv_json_path: str, role: str, candidate_id: str | None = None) -> None:
    rnd = LovelaceRound(cv_json_path, role, candidate_id=candidate_id)
    print("Designing your coding problem...")
    step = rnd.start()
    if step["status"] == "error":
        print("\nCould not start the coding round:", step["error"])
        return

    print_problem(step["problem"])
    detected = step.get("detected_languages") or []
    default = detected[0] if detected else "python"
    lang = normalize_language(
        input(f"\nChoose your language {list(SUPPORTED_LANGUAGES)} [default: {default}]: ")
        or default) or default

    while True:
        code = _read_code()
        if not code.strip():
            print("Nothing pasted - try again.")
            continue
        step = rnd.submit(code, lang)
        if step["status"] != "awaiting_submission":
            break
        print(f"\n  {step.get('message') or step.get('error') or ''}")
        if input("Type DONE to finish and get graded, or press Enter to resubmit: "
                 ).strip().lower() == "done":
            break

    print("\nGrading your submission...")
    step = rnd.finalize()
    while step["status"] == "awaiting_submission":     # grader hiccup: offer a retry
        print(f"\n  ! {step.get('error')}")
        if input("Press Enter to retry grading (or type Q to quit): ").strip().lower() == "q":
            return
        step = rnd.finalize()
    if step["status"] != "completed":
        print("Could not finalize:", step.get("error"))
        return
    print_result(step)
    print(f"\nSaved problem to: {step['problem_path']}")
    print(f"Saved result to:  {step['result_path']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Agent Lovelace: the coding round.")
    ap.add_argument("cv_json", help="parsed CV JSON file")
    ap.add_argument("role", help="role applied for")
    ap.add_argument("--candidate-id", default=None)
    a = ap.parse_args()
    run_cli(a.cv_json, a.role, a.candidate_id)