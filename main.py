from config.config import Config

if __name__ == '__main__':
    # Load config
    config = Config()

    # Download data

    # Filter data

    # Load commit code

    # Train/test split (before training embedding)

    # Embed code

    # Embed commit msg

    # Create batches
    # [X y] form / data with labels

    # For every batch (until # epoch):
    # .. encode code embedding (creates hidden state)
    # .. hidden state magic (h1, h2 -> plus/min/avg)
    # .. decode hidden states (creates text embedding (use attention??))
    # .. calculate loss with embedding of commit msg
    # .. backpropagate magic

    # PROFIT

    pass
