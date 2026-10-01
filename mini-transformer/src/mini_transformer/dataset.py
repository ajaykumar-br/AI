def create_training_data(token_ids, sequence_length):
    """
    Given a sequence of token IDs, create:

        input_ids
        target_ids

    Remember:
        input -> target should be shifted by one position.

    Example:

        input:  [I,  like, cats]
        target: [like, cats, <...>]
    """

    input_ids = []
    target_ids = []

    for i in range(len(token_ids) - sequence_length):
      input_ids.append(token_ids[i:i+sequence_length])
      target_ids.append(token_ids[i+1:i+sequence_length+1])

    return input_ids, target_ids