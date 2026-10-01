from mini_transformer.loss import calculate_loss

def train(model, input_ids, target_ids, optimizer, epochs):
    loss_history = []
    for epoch in range(epochs):

        # TODO
        # Forward pass
        logits = model(input_ids)

        # TODO
        # Calculate loss
        loss = calculate_loss(logits, target_ids)

        # TODO
        # Zero gradients
        optimizer.zero_grad()

        # TODO
        # Backward pass
        loss.backward()

        # TODO
        # Update weights
        optimizer.step()

        # TODO
        # Print loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1} / loss: {loss.item():.4f}")
        loss_history.append(loss.item())
    return loss_history