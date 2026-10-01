import torch

from mini_transformer.tokenizer import tokenize
from mini_transformer.vocabulary import (
    build_vocabulary,
    encode,
)
from mini_transformer.dataset import create_training_data
from mini_transformer.model import TinyLanguageModel
from mini_transformer.training import train

if __name__ == "__main__":
    text = """
    The cat sat on the mat.
    The dog sat on the floor.
    """

    sequence_length = 8
    embedding_dim = 32
    head_dim = 32

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = tokenize(text)

    # TODO
    # Build vocabulary

    stoi, itos = build_vocabulary(tokens)

    # TODO
    # Encode tokens
    
    token_ids = encode(tokens, stoi)

    # TODO
    # Create input/target sequences

    input_ids, target_ids = create_training_data(token_ids, sequence_length)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    # TODO
    # Make sure token IDs are torch.long
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    # TODO
    # Instantiate TinyLanguageModel
    model = TinyLanguageModel(vocab_size=len(stoi), embedding_dim=embedding_dim, head_dim=head_dim, sequence_length=sequence_length)

    # --------------------------------------------------------
    # Test forward pass
    # --------------------------------------------------------

    # TODO
    # Run one forward pass
    logits = model(input_ids)
    print("input shape:", input_ids.shape)
    print("logits shape:", logits.shape)

    # TODO
    # Print:
    #
    # input shape
    # logits shape
    #
    # You should understand why the logits
    # have the shape:
    #
    # (batch_size, sequence_length, vocab_size)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    # TODO
    # Create optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # TODO
    # Train the model
    train(model, input_ids, target_ids, optimizer, epochs = 100)