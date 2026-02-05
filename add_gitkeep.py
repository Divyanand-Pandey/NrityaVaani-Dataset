import os
from pathlib import Path
import argparse


def add_gitkeeps(root: Path):
    created = []
    if not root.exists():
        print(f"Path '{root}' does not exist")
        return created

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        gitkeep = p / ".gitkeep"
        if not gitkeep.exists():
            try:
                gitkeep.write_text("")
                created.append(str(gitkeep))
            except Exception as e:
                print(f"Failed to create {gitkeep}: {e}")
    return created


def main():
    p = argparse.ArgumentParser(description="Create .gitkeep in each folder under a root path")
    p.add_argument("root", nargs="?", default="final_dataset/val", help="Root folder to scan")
    args = p.parse_args()

    root = Path(args.root)
    created = add_gitkeeps(root)
    if created:
        print("Created .gitkeep files:")
        for c in created:
            print(" -", c)
    else:
        print("No new .gitkeep files created.")


if __name__ == "__main__":
    main()
