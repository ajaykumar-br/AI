def transcript_to_text(transcript):
    """
    Convert transcript snippets into one continuous text string.
    """

    text = " ".join(
        snippet.text
        for snippet in transcript
    )

    return text