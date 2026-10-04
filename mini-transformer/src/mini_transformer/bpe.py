import re
from collections import Counter

# Split text into chunks before BPE, so merges never cross word boundaries.
# The leading space stays attached to the word:
#   "the cat sat." -> ["the", " cat", " sat", "."]
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def get_pair_counts(word_counts):
    """
    How often each adjacent pair of IDs appears, weighted by word frequency.
    """

    pair_counts = Counter()
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
    return pair_counts


def merge_word(word, pair, new_id):
    """
    Replace every occurrence of `pair` in `word` with `new_id`.

        (104, 101, 108, 108, 111), (108, 108), 300 -> (104, 101, 300, 111)
    """

    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


class BPETokenizer:

    def __init__(self, merges=None):
        # (id_a, id_b) -> new_id, in the order they were learned.
        self.merges = dict(merges or {})
        self.vocab = self._build_vocab()
        self._cache = {}

    def _build_vocab(self):
        # IDs 0-255 are the raw bytes, so any text can be encoded
        # and there is no <UNK>.
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        """
        Learn vocab_size - 256 merges from text.
        """

        # Count each distinct chunk once instead of walking the whole corpus.
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        word_counts = {
            tuple(chunk.encode("utf-8")): count
            for chunk, count in chunk_counts.items()
        }

        self.merges = {}
        for new_id in range(256, vocab_size):
            pair_counts = get_pair_counts(word_counts)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges[best_pair] = new_id
            word_counts = {
                merge_word(word, best_pair, new_id): count
                for word, count in word_counts.items()
            }

            if verbose and (new_id - 255) % 250 == 0:
                self.vocab = self._build_vocab()
                print(f"merge {new_id - 255}: {self.vocab[new_id]!r} ({pair_counts[best_pair]} times)")

        self.vocab = self._build_vocab()
        self._cache = {}

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]

        word = tuple(chunk.encode("utf-8"))
        while len(word) >= 2:
            # Apply the earliest-learned merge first, same order as training.
            pair = min(
                zip(word, word[1:]),
                key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            word = merge_word(word, pair, self.merges[pair])

        self._cache[chunk] = list(word)
        return self._cache[chunk]

    def encode(self, text):
        token_ids = []
        for chunk in SPLIT_PATTERN.findall(text):
            token_ids.extend(self._encode_chunk(chunk))
        return token_ids

    def decode(self, token_ids):
        # A token can end halfway through a multi-byte character,
        # so join the bytes first and decode once.
        data = b"".join(self.vocab[i] for i in token_ids)
        return data.decode("utf-8", errors="replace")

    def token_strings(self, token_ids):
        """
        Each ID as its own string, for printing predictions.
        """

        return [self.vocab[i].decode("utf-8", errors="replace") for i in token_ids]
