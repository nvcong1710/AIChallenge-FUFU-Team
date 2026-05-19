"""
Extract Mermaid blocks from KIEN-TRUC-MERMAID.md, strip icons, render to PNG.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_FILE = ROOT / "KIEN-TRUC-MERMAID.md"
DIAGRAMS_DIR = ROOT / "diagrams"
SOURCES_DIR = DIAGRAMS_DIR / "sources"
DIAGRAMS_DIR.mkdir(exist_ok=True)
SOURCES_DIR.mkdir(exist_ok=True)

NAMES = [
    "01-doppelsearch",
    "02-vi-atiso",
    "03-naivenotnice",
    "04-newsinsight",
    "05-vizquest",
    "06-snapseek",
    "07-rapid",
    "08-tychevid",
    "09-dmar",
    "10-session-adapter",
    "11-temporal-event-graph",
    "12a-combo-architecture",
    "12b-combo-sequence",
]

# Unicode emoji ranges + arrows we want to strip
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def strip_icons(text: str) -> str:
    # Remove emojis
    text = EMOJI_RE.sub("", text)
    # Clean up double spaces left behind
    text = re.sub(r"  +", " ", text)
    # Trim leading/trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def extract_blocks(md: str) -> list[str]:
    pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
    return pattern.findall(md)


def rewrite_md_without_icons(md: str) -> str:
    """Strip icons ONLY inside mermaid blocks; keep section headers intact."""
    def repl(m: re.Match) -> str:
        return "```mermaid\n" + strip_icons(m.group(1)) + "```"
    return re.sub(r"```mermaid\n(.*?)```", repl, md, flags=re.DOTALL)


import shutil

MMDC = shutil.which("mmdc.cmd") or shutil.which("mmdc") or "mmdc"


def render(mmd_path: Path, png_path: Path) -> bool:
    cmd = [
        MMDC,
        "-i", str(mmd_path),
        "-o", str(png_path),
        "-b", "white",
        "-w", "1600",
        "-s", "2",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,
        )
        if result.returncode != 0:
            print(f"  FAIL: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def main() -> int:
    md = MD_FILE.read_text(encoding="utf-8")

    # 1. Rewrite MD with icons stripped inside mermaid blocks
    new_md = rewrite_md_without_icons(md)
    if new_md != md:
        MD_FILE.write_text(new_md, encoding="utf-8")
        print(f"Updated {MD_FILE.name} (icons stripped from mermaid blocks)")

    # 2. Extract clean blocks
    blocks = extract_blocks(new_md)
    print(f"Found {len(blocks)} mermaid blocks; expected {len(NAMES)}")
    if len(blocks) != len(NAMES):
        print("WARNING: count mismatch")

    # 3. Write .mmd and render PNG
    failed = []
    for name, block in zip(NAMES, blocks):
        mmd_path = SOURCES_DIR / f"{name}.mmd"
        png_path = DIAGRAMS_DIR / f"{name}.png"
        mmd_path.write_text(block, encoding="utf-8")
        print(f"Rendering {name}...")
        if not render(mmd_path, png_path):
            failed.append(name)

    print()
    print(f"Done. {len(NAMES) - len(failed)}/{len(NAMES)} rendered successfully.")
    if failed:
        print(f"Failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
