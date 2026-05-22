from pathlib import Path
from typing import Callable
import pymupdf4llm

class ResumeToMD:
    """
    A class to convert PDF resumes to Markdown format.
    """
    def __init__(self, src_dir: Path, dest_dir: Path):
        self.src_dir = src_dir
        self.dest_dir = dest_dir

    def convert_to_md(self, filename: Path) -> bool:
        src_file = filename
        relative_path = filename.relative_to(self.src_dir)
        dest_file = self.dest_dir / relative_path
        dest_file = dest_file.with_suffix(".md")
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {src_file} to Markdown format {dest_file}...")
        md = pymupdf4llm.to_markdown(src_file, header=False, footer=False)
        dest_file.write_bytes(md.encode())
        return True

    def process_resumes(self, converter: Callable[[Path], str]):
        files: list[Path] = []

        def process_directory(root_dir):
            for item in root_dir.iterdir():
                if item.is_file() and item.suffix == ".pdf":
                    converter(item)
                    files.append(item)
                elif item.is_dir():
                    process_directory(item)

        process_directory(self.src_dir)
        return files

if __name__ == "__main__":
    src_directory = Path("../../data/resumes/")
    dest_directory = Path("../../data/md/")
    resd = ResumeToMD(src_directory, dest_directory)
    resd.process_resumes(resd.convert_to_md)
