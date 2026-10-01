def build_vocabulary(tokens):
    stoi = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    itos = {
        0: "<PAD>",
        1: "<UNK>"
    }

    current_id = 2

    for token in tokens:
        if token not in stoi:
            stoi[token] = current_id
            itos[current_id] = token
            current_id += 1

    return stoi, itos


def encode(tokens, stoi):
    """
    Convert tokens into token IDs.

    Unknown tokens should map to <UNK>.
    """

    encoded = []
    for token in tokens:
      if token not in stoi:
        encoded.append(stoi["<UNK>"])
      else:
        encoded.append(stoi[token])

    return encoded


def decode(token_ids, itos):
    """
    Convert token IDs back into tokens.
    """

    decoded = []
    for id in token_ids:
      decoded.append(itos[id])

    return decoded