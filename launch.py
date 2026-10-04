"""Open the local analysis page, starting its server when needed."""

import subprocess
import json
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from optical_agent import __version__


BASE = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:7860"
URL = BASE_URL + "/agent"
GRAPH_PYTHON = Path(r'D:\CodexData\optical_agent\langgraph\venv\Scripts\python.exe')


def ready():
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=1) as response:
            info = json.load(response)
            return (response.status == 200 and info.get('app_version') == __version__
                    and ('--langgraph' not in sys.argv or info.get('langgraph_available') is True))
    except (OSError, TimeoutError, ValueError):
        return False


def main():
    if '--langgraph' in sys.argv and not GRAPH_PYTHON.exists():
        raise SystemExit('LangGraph environment missing. See docs/13_LANGGRAPH_RUNTIME.md')
    if not ready():
        log_dir = (Path(r'D:\CodexData\optical_agent\langgraph\runs') if GRAPH_PYTHON.exists()
                   else BASE.parent.parent / "work" / "ui_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "server.out.log").open("ab") as stdout, (log_dir / "server.err.log").open("ab") as stderr:
            subprocess.Popen(
                [str(GRAPH_PYTHON) if GRAPH_PYTHON.exists() else sys.executable, str(BASE / "app.py")],
                cwd=BASE,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        for _ in range(30):
            if ready():
                break
            time.sleep(1)
    if not ready():
        raise SystemExit("Could not start the analysis page. Check work/ui_logs/server.err.log")
    webbrowser.open(URL)
    print("Analysis page opened: " + URL)


if __name__ == "__main__":
    main()
