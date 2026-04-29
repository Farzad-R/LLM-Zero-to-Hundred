#!/usr/bin/env python3
"""
PostToolUse hook — runs pytest after any Python file edit in RAG-GPT.

Reads the Claude Code hook payload from stdin, checks whether the edited
file belongs to RAG-GPT, and if so runs the test suite. Outputs a JSON
systemMessage so Claude sees pass/fail on the next turn.

Invoked by settings.json:
  hooks > PostToolUse > matcher "Edit|Write" > command "python .claude/hooks/run_pytest.py"
"""
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return

    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith(".py"):
        return

    # Walk upward to find the RAG-GPT project root
    path = Path(file_path)
    rag_gpt_dir = next(
        (p for p in [path] + list(path.parents) if p.name == "RAG-GPT"),
        None,
    )
    if rag_gpt_dir is None or not (rag_gpt_dir / "tests").exists():
        return

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(rag_gpt_dir),
            capture_output=True,
            text=True,
            timeout=110,  # leave 10s margin below the 120s hook timeout
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"systemMessage": "[pytest] Timed out after 110s"}))
        return

    output = (result.stdout + result.stderr).strip()
    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(json.dumps({
        "systemMessage": f"[pytest] {status} (rc={result.returncode})\n\n{output}"
    }))


if __name__ == "__main__":
    main()
