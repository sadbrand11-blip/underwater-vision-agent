"""File/history audit: print file names and categories, never matching secrets."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ALLOW_KEY_FIXTURES={'tests/test_adaptive.py','tests/test_langgraph.py','tests/test_mcp.py'}
PATTERNS={
    'personal_path':re.compile(r'[A-Z]:[\\/]+Users[\\/]+(?:sol|[^\s<>/\\]+)',re.I),
    'credential':re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'populated_env_key':re.compile(r'(?m)^(?:DEEPSEEK_API_KEY|LLM_API_KEY)[ \t]*=[ \t]*[^\s#]+'),
}


def audit():
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
    names=tracked.stdout.decode().split('\0') if tracked.returncode==0 and tracked.stdout else [str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    errors=[];allowed=[]
    for name in filter(None,names):
        path=ROOT/name
        if path.name=='.env' or path.suffix in {'.pt','.pth','.sqlite3','.db','.zip'} or name.startswith('runs/'):
            errors.append({'file':name,'category':'excluded_artifact'})
        try:text=path.read_text(encoding='utf8')
        except (UnicodeDecodeError,OSError):continue
        for category,pattern in PATTERNS.items():
            if pattern.search(text):
                item={'file':name,'category':category}
                (allowed if category=='credential' and name in ALLOW_KEY_FIXTURES else errors).append(item)
    revisions=subprocess.run(['git','rev-list','--all'],cwd=ROOT,capture_output=True,text=True)
    commits=revisions.stdout.splitlines() if revisions.returncode==0 else []
    for commit in commits:
        raw=subprocess.run(['git','show','--format=fuller','--no-ext-diff',commit],cwd=ROOT,capture_output=True).stdout.decode('utf8','replace')
        author=raw.split('diff --git',1)[0]
        if re.search(r'(?m)^\s*(?:Author|Commit):.*<(?!\d+\+sadbrand11-blip@users\.noreply\.github\.com)[^>]+>',author):
            errors.append({'commit':commit,'category':'non_public_author_email'})
        files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT).decode().splitlines()
        for name in files:
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            try:text=blob.decode('utf8')
            except UnicodeDecodeError:continue
            for category,pattern in PATTERNS.items():
                if pattern.search(text) and not (category=='credential' and name in ALLOW_KEY_FIXTURES):
                    errors.append({'commit':commit,'file':name,'category':category})
    result={'files_checked':len(list(filter(None,names))),'commits_checked':len(commits),'errors':errors,'fabricated_fixture_allowlist':allowed,'status':'pass' if not errors else 'fail'}
    print(json.dumps(result,indent=2))
    return not errors


if __name__=='__main__':raise SystemExit(0 if audit() else 1)
