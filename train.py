import torch
from model.LossCompute import LossCompute
from model.utils import make_model, run_epoch

from config.config import Config

if __name__ == '__main__':
    # Load config
    config = Config()

    # Load the data
    trn_data = None
    eval_data = None

    # Construct the model
    # TODO: change the criterion
    criterion = torch.nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(config)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Set the parameters of the model to the GPU
    if config.USE_CUDA:
        model.cuda()

    for epoch in range(config.NUM_EPOCHS):
        print("Epoch %d" % epoch)

        # Train
        model.train()
        run_epoch(trn_data, model,
                  LossCompute(model.generator, criterion, optim))

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            run_epoch(eval_data, model,
                      LossCompute(model.generator, criterion, None))

            # TODO: encode/decode some eval data and analyse the output
