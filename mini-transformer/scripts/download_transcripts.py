from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi


VIDEOS_FILE = Path("data/raw/videos.txt")
OUTPUT_DIR = Path("data/raw/transcripts")

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    raise ValueError(f"Unsupported YouTube URL: {url}")


def download_transcripts():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = VIDEOS_FILE.read_text(encoding="utf-8").splitlines()

    api = YouTubeTranscriptApi()

    for url in urls:

        url = url.strip()

        if not url:
            continue

        video_id = extract_video_id(url)

        print(f"\nDownloading transcript: {video_id}")

        try:
            transcript = api.fetch(
                video_id,
                languages=["en"]
            )

            text = " ".join(
                snippet.text
                for snippet in transcript
            )

            output_path = OUTPUT_DIR / f"{video_id}.txt"

            output_path.write_text(
                text,
                encoding="utf-8"
            )

            print(f"Saved to: {output_path}")

        except Exception as e:
            print(f"FAILED: {video_id}")
            print(f"Reason: {e}")
            continue


if __name__ == "__main__":
    download_transcripts()