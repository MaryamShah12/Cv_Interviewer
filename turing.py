"""
Agent Turing
------------
1. Loads + validates the candidate CV JSON.
2. Generates all 10 MCQs (with answer key) in ONE LLM call and validates them.
3. Asks the candidate ONE question at a time using LangGraph `interrupt()`.
   The answer key never leaves the graph state while the quiz is running.
4. Scores the quiz (1 mark per question, 10 marks total) and saves results.

CLI usage:
    python turing.py <cv_json_path> "<role>"

Programmatic usage (e.g. from a web backend):
    quiz = TuringQuiz(cv_json_path, role)
    step = quiz.start()                 # {"status": "awaiting_answer", "question": ..., "options": ...}
    step = quiz.answer("b")             # next question ... or {"status": "completed", "score": 7, ...}
"""
import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

TOTAL_QUESTIONS = 10
MARKS_PER_QUESTION = 1
MAX_GENERATION_ATTEMPTS = 3
OPTION_KEYS = ("a", "b", "c", "d")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class MCQ(BaseModel):
    question: str = Field(min_length=10)
    option_a: str = Field(min_length=1)
    option_b: str = Field(min_length=1)
    option_c: str = Field(min_length=1)
    option_d: str = Field(min_length=1)
    correct_option: Literal["a", "b", "c", "d"]
    explanation: str = Field(min_length=1)

    @field_validator("correct_option", mode="before")
    @classmethod
    def _normalise_correct(cls, v):
        # LLMs sometimes return "A", "a)", "Option B" etc.
        if isinstance(v, str):
            m = re.search(r"[abcdABCD]", v.strip())
            if m and len(v.strip()) <= 10:
                return m.group(0).lower()
        return v


class MCQSet(BaseModel):
    role: str
    mcqs: list[MCQ]


def empty_mcq_set(role: str = "", confidence: str = "none") -> dict:
    return {"role": role, "mcqs": [], "_generation_status": confidence}


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class TuringState(TypedDict):
    cv_json_path: str
    role: str
    skills: list
    projects: list
    input_error: str | None
    llm_raw_output: str
    mcqs: dict | None              # validated MCQs + answer key (never shown to user mid-quiz)
    validation_error: str | None
    attempts: int
    # ---- quiz state ----
    current_index: int             # index of the question currently being asked
    answers: list                  # candidate's answers so far (letters)
    last_input_error: str | None   # set when the candidate typed an invalid answer
    score: int | None
    total: int
    review: list                   # per-question result, built at the end
    output_path: str | None        # MCQ bank file
    result_path: str | None        # score / review file


# --------------------------------------------------------------------------- #
# Node 1: load + validate CV
# --------------------------------------------------------------------------- #
def _read_text_fallback(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")  # also strips a BOM if present
    except UnicodeDecodeError:
        return raw.decode("cp1252")  # Windows ANSI fallback


def _as_str_list(value) -> list[str]:
    """Defensive: CV JSON may contain None, a string, or non-string items."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]


def node_load_validate(state: TuringState) -> dict:
    path = Path(state["cv_json_path"])
    role = (state.get("role") or "").strip()

    try:
        if not path.exists():
            raise ValueError("CV JSON file does not exist")
        cv = json.loads(_read_text_fallback(path))
        if not isinstance(cv, dict):
            raise ValueError("CV JSON root must be an object")
    except ValueError as e:  # JSONDecodeError is a subclass of ValueError
        return {"skills": [], "projects": [], "input_error": f"Cannot load CV JSON: {e}"}

    skills = cv.get("skills")
    skills = skills if isinstance(skills, dict) else {}
    technical = _as_str_list(skills.get("technical"))
    tools = _as_str_list(skills.get("tools"))

    projects = []
    raw_projects = cv.get("projects")
    for p in raw_projects if isinstance(raw_projects, list) else []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        desc = str(p.get("description") or "").strip()
        if name:
            projects.append({"name": name, "description": desc})

    merged_skills = list(dict.fromkeys(technical + tools))  # de-dupe, keep order

    if not role:
        return {"skills": [], "projects": [], "input_error": "No role selected."}
    if len(merged_skills) < 2:
        return {"skills": merged_skills, "projects": [],
                "input_error": "CV has fewer than 2 extractable skills "
                               "(skills.technical + skills.tools are empty)."}
    if not projects:
        return {"skills": merged_skills, "projects": [],
                "input_error": "CV has no extractable projects."}

    return {"skills": merged_skills, "projects": projects, "input_error": None}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are Agent Turing, a senior technical interviewer with 15+ years of
experience conducting rigorous interviews at top-tier engineering companies.

Your ONLY job: given a candidate's extracted CV skills, their projects, and the role they
applied for, generate exactly 10 multiple-choice questions (MCQs).

=== WHAT MAKES A GOOD QUESTION (READ CAREFULLY) ===
1. Questions must be INTERVIEW-LEVEL. They test analytical thinking, decision-making,
   trade-off analysis, debugging scenarios, system/design decisions, and real-world
   judgment. They must NOT be definitional ("What is X?", "When do we use Y?").
2. Every question MUST be grounded in the candidate's ACTUAL skills and projects from
   the CV below. If the CV mentions LangGraph and LangChain and a project "MCP Agent",
   ask the kind of question you would ask someone who has genuinely BUILT that project
   with those tools — e.g. how they would structure state, handle failures, choose
   between approaches, or debug a specific scenario.
3. The role shapes the ANGLE of the question, not the topic list. Same skills +
   different role = different style of question (e.g. AI Engineer -> agent architecture,
   prompt/state design; Backend Engineer -> concurrency, APIs, reliability).
4. Blend roughly: 60% scenario/design/trade-off questions, 40% applied-concept
   questions that require deeper-than-surface understanding.

=== FEW-SHOT EXAMPLE ===
Input:
  Role: AI Engineer
  Skills: LangGraph, LangChain, Python, Redis
  Projects: "MCP Agent" — a Model Context Protocol server exposing database tools to LLMs

BAD questions (never produce these):
  - "What is LangGraph?"  (definitional)
  - "Which library is used for building agents?"  (trivial recall)
  - "What does MCP stand for?"  (irrelevant trivia)

GOOD questions (this is the level expected):
  - "In your MCP Agent project, you expose database tools to an LLM. During a demo, the
     LLM calls the 'delete_table' tool after misreading the schema context. Which design
     change most directly prevents destructive tool misuse without breaking the agent's
     ability to perform read operations?"  (options: human-in-the-loop confirmation on
     destructive tools / hide schema from prompt / switch to a smaller model / disable
     tool calling entirely)  -> correct: human-in-the-loop confirmation
  - "Your LangGraph agent needs to remember conversation state across a server restart.
     Redis is already in your stack. Which approach preserves resumable agent state with
     the least rework?" (options around checkpointers vs global variables vs re-prompting)

Note how the good questions: reference the candidate's actual stack, describe a realistic
situation, have exactly one clearly best answer, and plausible but wrong distractors.

=== HARD RULES — VIOLATIONS MAKE THE OUTPUT USELESS ===
1. NEVER ask about a skill, tool, library, or concept that is NOT in the candidate's
   CV skills/projects. Do not invent skills. Do not pad with generic CS trivia.
2. NEVER hallucinate project details. If a project's description is thin, ask about the
   project NAME plus the candidate's stated skills only.
3. Exactly 10 MCQs. Each has exactly 4 options and exactly ONE correct option
   ("a", "b", "c" or "d"). Distractors must be plausible, not obviously wrong.
4. Every question is self-contained: a candidate who truly has the experience must be
   able to answer WITHOUT seeing their CV.
5. Output MUST be a single valid JSON object matching the schema. No markdown,
   no code fences, no commentary, no text before or after the JSON.
6. Options are shuffled by the program after generation. So NEVER use "all of the above",
   "none of the above", "both A and B" style options, and NEVER refer to options by
   letter (in the question, in other options, or in the explanation).
7. Spread the correct answers across a, b, c and d; do not favour one letter. The
   questions must be distinct from each other (no two on the same narrow point).

=== INPUT ===
Role: {role}
Candidate skills: {skills}
Candidate projects: {projects}

=== OUTPUT SCHEMA ===
{schema}"""


def build_prompt(skills: list, projects: list, role: str,
                 repair_hint: str | None = None) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    schema_json = json.dumps(MCQSet.model_json_schema(), indent=2)
    skills_str = ", ".join(skills)
    projects_str = json.dumps(projects, ensure_ascii=False)

    sys_msg = SystemMessage(content=SYSTEM_PROMPT.format(
        role=role, skills=skills_str, projects=projects_str, schema=schema_json))

    user = "Generate the 10 MCQs now."
    if repair_hint:
        user = (f"Your previous output failed validation with this error:\n"
                f"{repair_hint}\nFix ONLY the invalid parts. "
                f"Return the full corrected 10-MCQ JSON.\n\n" + user)
    return [sys_msg, HumanMessage(content=user)]


# --------------------------------------------------------------------------- #
# LLM output handling
# --------------------------------------------------------------------------- #
def _content_to_text(content) -> str:
    """Gemini/LangChain may return a str OR a list of content parts."""
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
    """Parse JSON from LLM output; tolerate code fences and surrounding chatter."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)  # happy path (JSON mode)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(cleaned[start:end + 1])


# --------------------------------------------------------------------------- #
# Node 2: generate all MCQs at once
# --------------------------------------------------------------------------- #
def node_generate_mcqs(state: TuringState, llm) -> dict:
    attempts = state.get("attempts", 0) + 1
    repair = state.get("validation_error") if attempts > 1 else None
    try:
        response = llm.invoke(build_prompt(
            state["skills"], state["projects"], state["role"], repair_hint=repair))
        raw = _content_to_text(response.content if hasattr(response, "content") else response)
        data = extract_json(raw)
        if isinstance(data, list):               # model returned a bare list of MCQs
            data = {"mcqs": data}
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a JSON object")
        data["role"] = state["role"]             # never trust the model for this field
        return {"llm_raw_output": raw, "mcqs": data,
                "attempts": attempts, "validation_error": None}
    except Exception as e:  # noqa: BLE001 - any LLM/parse error -> retry path
        # mcqs=None so a stale invalid set from a previous attempt is never reused
        return {"mcqs": None, "attempts": attempts,
                "validation_error": f"generation attempt {attempts} failed: {e}"}


# --------------------------------------------------------------------------- #
# Node 3: validate output (+ shuffle options to remove LLM position bias)
# --------------------------------------------------------------------------- #
_BAD_OPTION = re.compile(r"\b(all|none|both)\s+of\s+the\s+(above|following)\b", re.I)


def _shuffle_options(q: dict) -> dict:
    pairs = [(k, q[f"option_{k}"]) for k in OPTION_KEYS]
    correct_text = q[f"option_{q['correct_option']}"]
    random.shuffle(pairs)
    out = dict(q)
    for new_key, (_, text) in zip(OPTION_KEYS, pairs):
        out[f"option_{new_key}"] = text
        if text == correct_text:
            out["correct_option"] = new_key
    return out


def node_validate_output(state: TuringState) -> dict:
    if state.get("mcqs") is None:
        return {}  # generation itself failed; validation_error already set

    try:
        data = MCQSet.model_validate(state["mcqs"]).model_dump()
    except ValidationError as e:
        return {"mcqs": None, "validation_error": str(e)}

    errors = []
    if len(data["mcqs"]) != TOTAL_QUESTIONS:
        errors.append(f"expected exactly {TOTAL_QUESTIONS} MCQs, got {len(data['mcqs'])}")

    seen_questions = set()
    for i, q in enumerate(data["mcqs"], start=1):
        opts = [q[f"option_{k}"].strip().lower() for k in OPTION_KEYS]
        if len(set(opts)) != 4:
            errors.append(f"MCQ {i}: duplicate options")
        if any(_BAD_OPTION.search(o) for o in opts):
            errors.append(f"MCQ {i}: 'all/none/both of the above' options are not allowed")
        key = q["question"].strip().lower()
        if key in seen_questions:
            errors.append(f"MCQ {i}: duplicate question")
        seen_questions.add(key)

    if errors:
        return {"mcqs": None, "validation_error": "; ".join(errors)}

    data["mcqs"] = [_shuffle_options(q) for q in data["mcqs"]]
    return {"mcqs": data, "validation_error": None, "total": len(data["mcqs"]),
            "current_index": 0, "answers": [], "last_input_error": None}


# --------------------------------------------------------------------------- #
# Node 4: ask ONE question (human-in-the-loop via interrupt)
# --------------------------------------------------------------------------- #
def normalize_answer(raw) -> str | None:
    """Accept 'a', 'B', 'c)', '(d)', 'option b', or '1'-'4'. Return a/b/c/d or None."""
    s = str(raw).strip().lower()
    m = re.fullmatch(r"(?:option\s*)?\(?([a-d1-4])\)?[.):]?", s)
    if not m:
        return None
    ch = m.group(1)
    return OPTION_KEYS[int(ch) - 1] if ch.isdigit() else ch


def node_ask_question(state: TuringState) -> dict:
    from langgraph.types import interrupt

    mcqs = state["mcqs"]["mcqs"]
    idx = state["current_index"]
    q = mcqs[idx]

    # NOTE: only question + options go out. The answer key stays in graph state.
    payload = {
        "question_number": idx + 1,
        "total": len(mcqs),
        "question": q["question"],
        "options": {k: q[f"option_{k}"] for k in OPTION_KEYS},
        "error": state.get("last_input_error"),
    }

    # Graph pauses here. The value passed to Command(resume=...) is returned.
    raw = interrupt(payload)

    choice = normalize_answer(raw)
    if choice is None:
        # invalid input: stay on the same question and re-ask
        return {"last_input_error": f"Invalid answer {str(raw)!r}. Please reply with a, b, c or d."}
    return {"answers": list(state["answers"]) + [choice],
            "current_index": idx + 1,
            "last_input_error": None}


# --------------------------------------------------------------------------- #
# Node 5: score (1 mark per question)
# --------------------------------------------------------------------------- #
def node_score(state: TuringState) -> dict:
    mcqs = state["mcqs"]["mcqs"]
    review, score = [], 0
    for i, (q, ans) in enumerate(zip(mcqs, state["answers"]), start=1):
        ok = ans == q["correct_option"]
        score += MARKS_PER_QUESTION if ok else 0
        review.append({
            "question_number": i,
            "question": q["question"],
            "options": {k: q[f"option_{k}"] for k in OPTION_KEYS},
            "your_answer": ans,
            "correct_option": q["correct_option"],
            "is_correct": ok,
            "marks": MARKS_PER_QUESTION if ok else 0,
            "explanation": q["explanation"],
        })
    return {"score": score, "review": review}


# --------------------------------------------------------------------------- #
# Node 6: save (MCQ bank with key + quiz result)
# --------------------------------------------------------------------------- #
def node_save(state: TuringState) -> dict:
    out_dir = Path(os.getenv("TURING_OUTPUT_DIR", "generated_questions"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(state["cv_json_path"]).stem.replace("_cv", "")

    # --- MCQ bank (questions + answer key) ---
    bank = dict(state.get("mcqs") or empty_mcq_set(role=state.get("role", "")))
    if state.get("input_error"):
        bank["_input_error"] = state["input_error"]
    if state.get("validation_error"):
        bank["_generation_error"] = state["validation_error"]
    out_path = out_dir / f"{stem}_turing_mcqs.json"
    out_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")

    update = {"output_path": str(out_path)}

    # --- Quiz result (only if the candidate actually took the quiz) ---
    if state.get("score") is not None:
        result = {
            "role": state["role"],
            "score": state["score"],
            "total_marks": state["total"] * MARKS_PER_QUESTION,
            "review": state["review"],
        }
        result_path = out_dir / f"{stem}_turing_result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        update["result_path"] = str(result_path)
    return update


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(llm, checkpointer=None):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(TuringState)
    g.add_node("load_validate", node_load_validate)
    g.add_node("generate_mcqs", lambda s: node_generate_mcqs(s, llm))
    g.add_node("validate_output", node_validate_output)
    g.add_node("ask_question", node_ask_question)
    g.add_node("score", node_score)
    g.add_node("save", node_save)

    def route_after_input_check(state):
        # NO LLM CALL at all if input validation failed
        return "save" if state.get("input_error") else "generate_mcqs"

    def route_after_validation(state):
        if not state.get("validation_error"):
            return "ask_question"
        if state.get("attempts", 0) < MAX_GENERATION_ATTEMPTS:
            return "generate_mcqs"
        return "save"  # give up; save writes the error info

    def route_after_question(state):
        return "ask_question" if state["current_index"] < state["total"] else "score"

    g.add_edge(START, "load_validate")
    g.add_conditional_edges("load_validate", route_after_input_check,
                            {"save": "save", "generate_mcqs": "generate_mcqs"})
    g.add_edge("generate_mcqs", "validate_output")
    g.add_conditional_edges("validate_output", route_after_validation,
                            {"ask_question": "ask_question",
                             "generate_mcqs": "generate_mcqs", "save": "save"})
    g.add_conditional_edges("ask_question", route_after_question,
                            {"ask_question": "ask_question", "score": "score"})
    g.add_edge("score", "save")
    g.add_edge("save", END)

    # interrupt() REQUIRES a checkpointer. Swap MemorySaver for SqliteSaver /
    # PostgresSaver if the quiz must survive a server restart.
    return g.compile(checkpointer=checkpointer or MemorySaver())


def get_llm():
    """Gemini Flash. API key from GEMINI_API_KEY env var."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
        max_tokens=8000,
        # Gemini's native JSON mode ("response_format" is an OpenAI-only kwarg)
        response_mime_type="application/json",
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class TuringQuiz:
    """Drives the graph: start() -> answer() -> answer() ... -> completed."""

    def __init__(self, cv_json_path: str, role: str, llm=None, checkpointer=None,
                 thread_id: str | None = None):
        self.graph = build_graph(llm or get_llm(), checkpointer)
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
        self._init: TuringState = {
            "cv_json_path": cv_json_path, "role": role,
            "skills": [], "projects": [],
            "input_error": None, "llm_raw_output": "", "mcqs": None,
            "validation_error": None, "attempts": 0,
            "current_index": 0, "answers": [], "last_input_error": None,
            "score": None, "total": 0, "review": [],
            "output_path": None, "result_path": None,
        }

    def start(self) -> dict:
        from langgraph.errors import GraphInterrupt  # noqa: F401 (older versions may raise)
        try:
            self.graph.invoke(self._init, self.config)
        except GraphInterrupt:
            pass
        return self._status()

    def answer(self, choice: str) -> dict:
        from langgraph.errors import GraphInterrupt  # noqa: F401
        from langgraph.types import Command
        try:
            self.graph.invoke(Command(resume=choice), self.config)
        except GraphInterrupt:
            pass
        return self._status()

    def _status(self) -> dict:
        snap = self.graph.get_state(self.config)
        if snap.next:  # paused at an interrupt
            for task in snap.tasks:
                if task.interrupts:
                    return {"status": "awaiting_answer", **task.interrupts[0].value}
        v = snap.values
        if v.get("score") is not None:
            total = v["total"] * MARKS_PER_QUESTION
            return {"status": "completed", "score": v["score"], "total_marks": total,
                    "percentage": round(100 * v["score"] / total, 1),
                    "review": v["review"],
                    "output_path": v.get("output_path"), "result_path": v.get("result_path")}
        return {"status": "error",
                "error": v.get("input_error") or v.get("validation_error") or "Unknown error",
                "output_path": v.get("output_path")}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_cli(cv_json_path: str, role: str) -> None:
    quiz = TuringQuiz(cv_json_path, role)
    print("Generating your questions...")
    step = quiz.start()

    while step["status"] == "awaiting_answer":
        if step.get("error"):
            print(f"\n  ! {step['error']}")
        print(f"\nQuestion {step['question_number']}/{step['total']}  (1 mark)")
        print(step["question"])
        for k, text in step["options"].items():
            print(f"  {k.upper()}) {text}")
        step = quiz.answer(input("Your answer (a/b/c/d): "))

    if step["status"] == "error":
        print("\nCould not run the quiz:", step["error"])
        return

    print(f"\n=== RESULT: {step['score']} / {step['total_marks']} "
          f"({step['percentage']}%) ===")
    for r in step["review"]:
        mark = "✔" if r["is_correct"] else "✘"
        print(f"\nQ{r['question_number']} {mark}  your answer: {r['your_answer'].upper()}"
              f"  correct: {r['correct_option'].upper()}")
        if not r["is_correct"]:
            print(f"   Why: {r['explanation']}")
    print(f"\nSaved questions to: {step['output_path']}")
    print(f"Saved result to:    {step['result_path']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit('usage: python turing.py <cv_json_path> "<role>"')
    run_cli(sys.argv[1], sys.argv[2])