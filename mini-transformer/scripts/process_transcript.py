from pathlib import Path


INPUT_DIR = Path("data/raw/transcripts")
OUTPUT_DIR = Path("data/processed")


def clean_text(text):
    # TODO:
    # 1. Normalize whitespace
    # 2. Remove unnecessary characters
    # 3. Return cleaned text

    text = text.split()
    step1 = " ".join(text) 

    return step1


def process_transcripts():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transcript_files = INPUT_DIR.glob("*.txt")

    for input_path in transcript_files:

        print(f"Processing: {input_path.name}")

        text = input_path.read_text(encoding="utf-8")

        cleaned_text = clean_text(text)

        output_path = OUTPUT_DIR / input_path.name

        output_path.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    process_transcripts()