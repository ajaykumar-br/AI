from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/processed/corpus.txt")


def build_corpus():

    # TODO:
    # 1. Find all .txt files in INPUT_DIR
    # 2. Read each transcript
    # 3. Combine them into one string
    # 4. Write the result to OUTPUT_FILE
    
    transcript_files = INPUT_DIR.glob("*.txt")

    transcripts = []

    for file_path in transcript_files:

        text = file_path.read_text(encoding="utf-8")

        transcripts.append(text)

    corpus = " ".join(transcripts)

    OUTPUT_FILE.write_text(
        corpus,
        encoding="utf-8"
    )


if __name__ == "__main__":
    build_corpus()