import torch

from config.config import Config

if __name__ == '__main':
    # Load config
    config = Config()

    # Load test data
    tst_data = None

    # Load the model
    path = config.LOAD_PATH
    model = None

    for tst in enumerate(tst_data):
        model.eval()
        with torch.no_grad():
            # Create the hidden states
            model.encode()

            # Decode to a sentence
            for i in range(config.DEC_MAX_LEN):
                model.decode()
