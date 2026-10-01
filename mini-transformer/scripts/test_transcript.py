from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from mini_transformer.cleaning import transcript_to_text

video_id = "_aw32rFL680"

api = YouTubeTranscriptApi()

transcript = api.fetch(video_id, languages=["en"])

text = transcript_to_text(transcript)

output_path = Path("data/raw/transcripts") / f"{video_id}.txt"

output_path.write_text(
    text,
    encoding="utf-8"
)

print(f"Saved to: {output_path}")