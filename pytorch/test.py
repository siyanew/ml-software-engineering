import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
from parse_config import ConfigParser

# Fix GPU problems by first calling current_device()
# https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()


def main(config):
    tensorboard = SummaryWriter(config.log_dir_test)
    logger = config.get_logger('test')

    # setup data_loader instances

    if config['data_loader']['iterator']:
        loaders = config.init_obj('data_loader', module_data)
        data_loader = loaders.split_test()
    else:
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

    # build model architecture
    model = config.init_obj('arch')
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss']['function'])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # set the padding index in the criterion such that we ignore pad tokens
    if config['loss']['padding_idx']:
        loss_fn = loss_fn(loaders.TRG.vocab.stoi['<pad>'])

    if 'packed' in config['arch'] and config['arch']['packed']:
        model.set_tokens(loaders.SRC.vocab.stoi['<pad>'],
                         loaders.TRG.vocab.stoi['<sos>'],
                         loaders.TRG.vocab.stoi['<eos>'])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0

    # print("Computing Loss...")
    # with torch.no_grad():
    #     for i, batch in enumerate(tqdm(data_loader)):
    #         output, target = model.process_batch(batch)
    #
    #         # computing loss, metrics on test set
    #         loss = loss_fn(output, target)
    #         batch_size = output.shape[0]
    #         total_loss += loss.item() * batch_size
    #
    # n_samples = len(data_loader)
    # print(n_samples)
    # avg_loss = total_loss / n_samples
    # log = {
    #     'loss':       avg_loss,
    #     'perplexity': math.exp(avg_loss)
    # }
    # logger.info(log)

    fpath = config['inference']['file']
    print(f"Inference on {fpath}")

    source_file = open(fpath, encoding='utf-8')
    target_file = open(f"{fpath}_pred", "a+")
    spacy_de = spacy.load('de')
    SRC = loaders.SRC
    TRG = loaders.TRG

    for idx, line in enumerate(tqdm(source_file)):
        # Strip newline characters
        line = line.strip()

        # Process from source line to target line
        tokenized_sentence = spacy_de.tokenizer(line)
        tokenized_sentence = ['<sos>'] + [t.text.lower() for t in tokenized_sentence] + ['<eos>']
        numericalized = [SRC.vocab.stoi[t] for t in tokenized_sentence]
        sentence_length = torch.LongTensor([len(numericalized)]).to(device)
        tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
        translation_tensor_logits, attention = model(tensor, sentence_length, None, 0)
        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
        translation = [TRG.vocab.itos[t] for t in translation_tensor]
        translation, attention = translation[1:], attention[1:]

        trg_pred = " ".join(translation)

        # Save results to tensorboard
        tensorboard.add_text(f"{idx}/src", line)
        tensorboard.add_text(f"{idx}/trg", trg_pred)

        # Visualize the attention
        att_fig = display_attention(line, translation, attention)
        tensorboard.add_figure(f"{idx}/attention", att_fig)
        plt.close(att_fig)

        target_file.write(trg_pred)
        target_file.write("\n")

    source_file.close()
    target_file.close()


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return fig


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
