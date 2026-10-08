"""
Interview Pipeline
------------------
Runs the whole screening flow in one go:

    CV file (.pdf/.docx)
      -> [1] CV parser   (test.py)       CV text  -> structured JSON       (LLM, no candidate input)
      -> [2] Agent Turing (turing.py)    10 technical MCQs from the CV      (candidate answers)
      -> [3] Agent Socrates (Socrates.py) 10 IQ / analytical MCQs from bank (candidate answers, 10 marks)
      -> [4] Agent Lovelace (lovelace.py) role-grounded coding problem      (candidate codes, 20 marks)
      -> pipeline_results/<candidate_id>_pipeline.json   (combined scores, 40 marks)

The four agent files are imported UNMODIFIED from the same folder (files are matched by name,
case-insensitively: test.py / cv_parser.py, turing.py, Socrates.py, lovelace.py).

Failure policy
  * CV cannot be parsed          -> pipeline stops (nothing to build the later rounds from).
  * Any agent round fails (LLM down, bad CV data ...) -> recorded, the next round still runs.
  * Lovelace fails               -> recorded; the summary is saved with whatever completed.

Environment (.env works): GROQ_API_KEY (CV parser), GEMINI_API_KEY (Turing),
GEMINI_API_KEY (Lovelace), optional SOCRATES_BANK_PATH,
PIPELINE_OUTPUT_DIR (default: pipeline_results).

CLI usage:
    python pipeline.py <cv_file> [--role "AI Engineer"] [--candidate-id ID] [--bank PATH]

Programmatic usage (e.g. from a FastAPI backend):
    pipe = InterviewPipeline("cv.pdf", "AI Engineer")
    step = pipe.start()       # {"status": "awaiting_answer", "stage": "turing", "question": ...}
    step = pipe.answer("b")   # ... "stage" switches to "socrates" by itself ...
    # coding round:
    step = pipe.submit_code("def solve(...): ...", language="python")  # feedback, resubmit freely
    step = pipe.finalize_coding()   # grades the round, moves to the summary
    # finally: {"status": "completed", "stage": "done", "summary": {...}, "summary_path": ...}
"""
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
DEFAULT_BANK_NAME = "socrates_iq_bank.json"
SUPPORTED_CV_TYPES = (".pdf", ".docx")

STAGE_TITLES = {
    "turing": "ROUND 1 - TECHNICAL (Agent Turing)",
    "socrates": "ROUND 2 - IQ & ANALYTICAL (Agent Socrates)",
    "lovelace": "ROUND 3 - CODING (Agent Lovelace)",
}
# The order rounds run in. A failed round is recorded and the next one still runs.
STAGE_ORDER = ("turing", "socrates", "lovelace")


# --------------------------------------------------------------------------- #
# Loading the three agent files
# --------------------------------------------------------------------------- #
def _load_agent(alias: str, *filenames: str):
    """Import a sibling .py file by path (name match is case-insensitive).

    Done by path on purpose: the CV parser is called `test.py`, which would clash with
    Python's built-in `test` package, and `Socrates.py` has a capital S.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    wanted = {n.lower() for n in filenames}
    for folder in (HERE, Path.cwd()):
        for f in sorted(folder.glob("*.py")):
            if f.name.lower() in wanted and f.resolve() != Path(__file__).resolve():
                spec = importlib.util.spec_from_file_location(alias, f)
                module = importlib.util.module_from_spec(spec)
                sys.modules[alias] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(alias, None)
                    raise
                return module
    raise ImportError(f"Could not find {' or '.join(filenames)} next to pipeline.py")


def _agents():
    return (_load_agent("cv_parser", "cv_parser.py", "Cvparser.py"),
            _load_agent("turing", "turing.py"),
            _load_agent("socrates", "socrates.py"),
            _load_agent("lovelace", "lovelace.py"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip()).strip("._")
    return stem or "candidate"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # a reader never sees a half-written file


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _default_bank_path() -> str | None:
    """Env var wins; else the bank sitting next to this file; else Socrates' own default."""
    if os.getenv("SOCRATES_BANK_PATH"):
        return os.environ["SOCRATES_BANK_PATH"]
    beside = HERE / DEFAULT_BANK_NAME
    return str(beside) if beside.exists() else None


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class InterviewPipeline:
    """CV parse -> Turing -> Socrates -> Lovelace. Drive it with start(), answer()
    (MCQ rounds), submit_code() / finalize_coding() (coding round)."""

    def __init__(self, cv_path: str, role: str, candidate_id: str | None = None,
                 bank_path: str | None = None, parser_llm=None, turing_llm=None,
                 lovelace_llm=None, checkpointer=None, thread_id: str | None = None,
                 continue_after_turing_error: bool = True):
        self.cv_path = Path(cv_path)
        self.role = (role or "").strip()
        self.candidate_id = (candidate_id or self.cv_path.stem).strip()
        self.bank_path = bank_path or _default_bank_path()
        self.continue_after_turing_error = continue_after_turing_error

        self._parser_llm, self._turing_llm = parser_llm, turing_llm  # None -> agent's get_llm()
        self._lovelace_llm = lovelace_llm
        self._code_language = "python"  # candidate's coding language; set via submit_code()
        self._checkpointer = checkpointer
        self._thread = thread_id or uuid.uuid4().hex

        self.stage = "new"           # new | turing | socrates | done | failed
        self.results: dict = {}      # per-stage outcome, filled as stages finish
        self.summary: dict | None = None
        self._last: dict = {}        # latest payload returned to the caller
        self._cv_json_path: str | None = None
        self._turing = self._socrates = self._lovelace = None
        self._cv_mod = self._turing_mod = self._socrates_mod = self._lovelace_mod = None

    # ------------------------------------------------------------------ public
    def start(self) -> dict:
        """Parse the CV, build the technical round and return its first question."""
        if self.stage != "new":      # never restart a running / finished pipeline
            return self._last

        # Cheap checks first, before any LLM call is paid for.
        if not self.role:
            return self._fatal("No role selected.")
        if not self.candidate_id:
            return self._fatal("No candidate id.")
        if not self.cv_path.is_file():
            return self._fatal(f"CV file not found: {self.cv_path}")
        if self.cv_path.suffix.lower() not in SUPPORTED_CV_TYPES:
            return self._fatal(f"Unsupported CV type '{self.cv_path.suffix}'. "
                               f"Only {', '.join(SUPPORTED_CV_TYPES)} allowed.")

        try:
            self._cv_mod, self._turing_mod, self._socrates_mod, self._lovelace_mod = _agents()
        except Exception as e:  # missing file, missing dependency (e.g. PyMuPDF) ...
            return self._fatal(f"Could not load the agent files: {_err(e)}")

        # Build both LLM clients up front so a missing API key fails NOW, not after the
        # candidate has already finished the first round.
        try:
            if self._parser_llm is None:
                self._parser_llm = self._cv_mod.get_llm()
            if self._turing_llm is None:
                self._turing_llm = self._turing_mod.get_llm()
            if self._lovelace_llm is None:
                self._lovelace_llm = self._lovelace_mod.get_llm()
        except Exception as e:  # KeyError for a missing key, ImportError for a missing package
            return self._fatal(f"LLM is not configured ({_err(e)}). "
                               f"Set GROQ_API_KEY (CV parser), GEMINI_API_KEY (Turing) "
                               f"and GEMINI_API_KEY (Lovelace).")

        error = self._parse_cv()
        if error:
            return self._fatal(error)
        return self._begin_stage(STAGE_ORDER[0])

    def answer(self, choice: str) -> dict:
        """Answer the current MCQ question. Moves between rounds automatically."""
        if self.stage == "new":
            return {"status": "error", "stage": "new", "error": "Pipeline has not been started."}
        if self.stage == "lovelace":
            return {"status": "error", "stage": "lovelace",
                    "error": "This is the coding round: submit code with submit_code(), "
                             "not answer()."}
        quiz = {"turing": self._turing, "socrates": self._socrates}.get(self.stage)
        if quiz is None:             # finished or failed: nothing to answer
            return self._last
        try:
            step = quiz.answer(choice)
        except Exception as e:       # never let one agent's crash lose the other's results
            step = {"status": "error", "error": _err(e)}
        return self._handle(self.stage, step)

    def submit_code(self, code: str, language: str | None = None) -> dict:
        """Submit code for the Lovelace coding round. Resubmission is allowed;
        call finalize_coding() when the candidate is done."""
        if self.stage != "lovelace" or self._lovelace is None:
            return {"status": "error", "stage": self.stage,
                    "error": "Not in the coding round."}
        if language:
            norm = self._lovelace_mod.normalize_language(language)
            if norm is None:
                return {"status": "error", "stage": "lovelace",
                        "error": f"Unsupported language {language!r}."}
            self._code_language = norm
        try:
            step = self._lovelace.submit(code, self._code_language)
        except Exception as e:
            step = {"status": "error", "error": _err(e)}
        return self._handle("lovelace", step)

    def finalize_coding(self) -> dict:
        """End the coding round: grade the best submission and move on."""
        if self.stage != "lovelace" or self._lovelace is None:
            return {"status": "error", "stage": self.stage,
                    "error": "Not in the coding round."}
        try:
            step = self._lovelace.finalize()
        except Exception as e:
            step = {"status": "error", "error": _err(e)}
        return self._handle("lovelace", step)

    # ------------------------------------------------------------------ stage 1
    def _parse_cv(self) -> str | None:
        """Run the CV parser. Returns an error message, or None on success."""
        try:
            out = self._cv_mod.parse_cv(str(self.cv_path), self._parser_llm)
        except Exception as e:
            return f"CV parsing crashed: {_err(e)}"

        cv, path = out.get("cv_json"), out.get("output_path")
        if out.get("extraction_error"):
            return f"Could not read the CV: {out['extraction_error']}"
        if not isinstance(cv, dict) or out.get("validation_error"):
            return f"Could not parse the CV into JSON: {out.get('validation_error') or 'no output'}"
        if str(cv.get("parse_confidence", "")).lower() == "none":
            return "The uploaded file does not look like a CV (parse confidence: none)."

        self._cv_json_path = path
        self.results["cv_parse"] = {"status": "completed", "output_path": path,
                                    "parse_confidence": cv.get("parse_confidence")}
        return None

    # ------------------------------------------------------------------ stages 2-4
    def _begin_stage(self, stage: str) -> dict:
        """Start one agent round and feed its first step into the state machine."""
        try:
            if stage == "turing":
                self._turing = self._turing_mod.TuringQuiz(
                    self._cv_json_path, self.role, llm=self._turing_llm,
                    checkpointer=self._checkpointer, thread_id=f"{self._thread}-turing")
                step = self._turing.start()
            elif stage == "socrates":
                self._socrates = self._socrates_mod.SocratesQuiz(
                    self.candidate_id, bank_path=self.bank_path,
                    checkpointer=self._checkpointer, thread_id=f"{self._thread}-socrates")
                step = self._socrates.start()
            elif stage == "lovelace":
                self._lovelace = self._lovelace_mod.LovelaceRound(
                    self._cv_json_path, self.role, candidate_id=self.candidate_id,
                    llm=self._lovelace_llm,
                    checkpointer=self._checkpointer, thread_id=f"{self._thread}-lovelace")
                step = self._lovelace.start()
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown stage {stage!r}")
        except Exception as e:
            step = {"status": "error", "error": _err(e)}
        return self._handle(stage, step)

    def _next_stage(self, stage: str) -> str | None:
        idx = STAGE_ORDER.index(stage)
        return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None

    # ------------------------------------------------------------------ plumbing
    _KEEP = {
        "turing": ("score", "total_marks", "percentage", "result_path"),
        "socrates": ("score", "total_marks", "percentage", "category_scores",
                     "difficulty_scores", "result_path"),
        "lovelace": ("score", "total_marks", "percentage", "test_marks",
                     "review_marks", "attempts_used", "result_path", "problem_path"),
    }

    def _handle(self, stage: str, step: dict) -> dict:
        """Central state machine: react to whatever an agent just reported."""
        status = step.get("status")

        if status in ("awaiting_answer", "awaiting_submission"):
            self.stage = stage
            self._last = {**step, "stage": stage}
            return self._last

        if status == "completed":
            record = {"status": "completed", **{k: step.get(k) for k in self._KEEP[stage]}}
            if stage == "turing":
                record["questions_path"] = step.get("output_path")  # MCQ bank incl. answer key
        else:
            record = {"status": "error", "error": step.get("error") or "Unknown error"}
            record.update({k: step[k] for k in ("result_path", "output_path") if step.get(k)})
        self.results[stage] = record

        nxt = self._next_stage(stage)
        if nxt and (status == "completed" or self.continue_after_turing_error):
            return self._begin_stage(nxt)
        return self._finalize()

    def _fatal(self, message: str) -> dict:
        self.stage = "failed"
        self.results["cv_parse"] = {"status": "error", "error": message}
        return self._finalize(fatal=message)

    def _finalize(self, fatal: str | None = None) -> dict:
        done = [r for r in self.results.values() if r.get("status") == "completed" and "score" in r]
        stages_ok = all(self.results.get(s, {}).get("status") == "completed"
                        for s in ("cv_parse",) + STAGE_ORDER)
        self.summary = {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "cv_path": str(self.cv_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if fatal else ("completed" if stages_ok else "partial"),
            "all_stages_completed": stages_ok,
            "total_score": sum(r["score"] for r in done),
            "total_marks": sum(r["total_marks"] for r in done),   # of the rounds that completed
            "stages": self.results,
        }

        out_dir = Path(os.getenv("PIPELINE_OUTPUT_DIR", "pipeline_results"))
        summary_path = out_dir / f"{_safe_stem(self.candidate_id)}_pipeline.json"
        try:
            _write_json_atomic(summary_path, self.summary)
            path = str(summary_path)
        except OSError:
            path = None

        if not fatal:
            self.stage = "done"
            self._last = {"status": "completed", "stage": "done",
                          "summary": self.summary, "summary_path": path}
        else:
            self._last = {"status": "error", "stage": "cv_parsing",
                          "error": fatal, "summary_path": path}
        return self._last


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pretty(text: str) -> str:
    return str(text).replace("_", " ").title()


def _banner(title: str) -> None:
    print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)


def _read_code_block() -> str:
    """Read multi-line code from the terminal (END on its own line submits)."""
    print("Paste your code, then type END on its own line and press Enter.")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _show_problem(problem: dict) -> None:
    print(f"\n{problem['title']}  ({problem['total_marks']} marks)")
    print("-" * 62)
    print(problem["statement"])
    print("\nExamples:")
    for i, e in enumerate(problem["examples"], 1):
        print(f"  {i}. input:  {e['input']}")
        print(f"     output: {e['output']}")
        if e.get("explanation"):
            print(f"     why:    {e['explanation']}")
    if problem.get("constraints"):
        print("\nConstraints:")
        for c in problem["constraints"]:
            print(f"  - {c}")
    print(f"\nSuggested time: ~{problem['time_estimate_min']} minutes")
    print(f"Marking: {problem['marking_scheme']}")


# (No code is executed: an LLM judge reads the candidate's logic, forgiving
# trivial syntax slips like a missing colon or semicolon.)


def _stage_line(label: str, res: dict | None) -> str:
    if not res:
        return f"  {label}: not run"
    if res.get("status") != "completed":
        return f"  {label}: unavailable ({res.get('error', 'unknown error')})"
    return f"  {label}: {res['score']} / {res['total_marks']} ({res['percentage']}%)"


def run_cli(cv_path: str, role: str | None = None, candidate_id: str | None = None,
            bank_path: str | None = None, **pipeline_kwargs) -> int:
    try:
        role = (role or input("Role you are applying for: ")).strip()
        pipe = InterviewPipeline(cv_path, role, candidate_id=candidate_id,
                                 bank_path=bank_path, **pipeline_kwargs)
        print("\nReading your CV and preparing the technical round "
              "(this can take a little while)...")
        step, shown, lang_chosen = pipe.start(), None, False

        while step["status"] in ("awaiting_answer", "awaiting_submission"):
            if step["stage"] != shown:
                just_done = shown
                _banner(STAGE_TITLES[step["stage"]])
                shown = step["stage"]
                if just_done == "turing":
                    print("\nTechnical round finished.")
                    print(_stage_line("Technical score", pipe.results.get("turing")))
                elif just_done == "socrates":
                    print("\nIQ / Analytical round finished.")
                    print(_stage_line("IQ score", pipe.results.get("socrates")))
            if step.get("error"):
                print(f"\n  ! {step['error']}")

            if step["status"] == "awaiting_submission":
                # ---------------- coding round ----------------
                if not lang_chosen:
                    _show_problem(step["problem"])
                    detected = step.get("detected_languages") or []
                    langs = step["supported_languages"]
                    default = detected[0] if detected else "python"
                    lang = (input(f"\nChoose your language {langs} "
                                  f"[default: {default}]: ") or default).strip()
                    if pipe._lovelace_mod.normalize_language(lang) is None:
                        print(f"Unknown language {lang!r}; using {default}.")
                        lang = default
                    pipe._code_language = pipe._lovelace_mod.normalize_language(lang)
                    lang_chosen = True
                code = _read_code_block()
                if not code.strip():
                    print("Nothing pasted - try again.")
                    continue
                print("\nSaving your submission...")
                step = pipe.submit_code(code)
                if step["status"] == "awaiting_submission":
                    print(f"\n  {step.get('message') or ''}")
                    nxt = input("\nPress Enter to try again, or type DONE to finish: ")
                    if nxt.strip().lower() == "done":
                        print("\nGrading your best submission...")
                        step = pipe.finalize_coding()
                continue

            # ---------------- MCQ rounds ----------------
            tag = f"  [{_pretty(step['category'])}]" if step.get("category") else ""
            print(f"\nQuestion {step['question_number']}/{step['total']}{tag}"
                  f"  ({step.get('marks', 1)} mark)")
            print(step["question"])
            for key, text in step["options"].items():
                print(f"  {key.upper()}) {text}")
            step = pipe.answer(input(f"Your answer ({'/'.join(step['options'])}): "))

    except (KeyboardInterrupt, EOFError):
        print("\n\nInterview cancelled.")
        return 130

    if step["status"] == "error":
        print(f"\nCould not start the interview: {step['error']}")
        if step.get("summary_path"):
            print(f"Details saved to: {step['summary_path']}")
        return 1

    s = step["summary"]
    _banner("FINAL RESULT")
    print(_stage_line("Technical (Turing)", s["stages"].get("turing")))
    print(_stage_line("IQ / Analytical (Socrates)", s["stages"].get("socrates")))
    print(_stage_line("Coding (Lovelace)", s["stages"].get("lovelace")))
    print(f"\n  TOTAL: {s['total_score']} / {s['total_marks']}")
    if not s["all_stages_completed"]:
        print("  (some rounds could not be completed - see the summary file)")
    print(f"\nSaved summary to: {step['summary_path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Run CV parsing -> Turing -> Socrates -> Lovelace in one go.")
    p.add_argument("cv_file", help="candidate CV (.pdf or .docx)")
    p.add_argument("--role", help="role applied for (asked interactively if omitted)")
    p.add_argument("--candidate-id", help="defaults to the CV file name")
    p.add_argument("--bank", help="Socrates question bank JSON "
                                  f"(default: SOCRATES_BANK_PATH or ./{DEFAULT_BANK_NAME})")
    a = p.parse_args(argv)
    return run_cli(a.cv_file, a.role, a.candidate_id, a.bank)


if __name__ == "__main__":
    sys.exit(main())