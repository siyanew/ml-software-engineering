import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
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
        # data_loader = loaders.split_test()
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

    # print("Computing Loss...")
    # total_loss = 0.0
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

    # Set up the translate function
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    SRC = loaders.SRC
    TRG = loaders.TRG
    translate = get_translation_fn(model, device, SRC, TRG)

    # Blue score init
    hypotheses = list()
    references = list()
    # adds epsilon counts
    smoother = SmoothingFunction().method1

    files = config['inference']

    logger.info(f"Starting inference")
    for key, value in files.items():
        t, file = str(key).split('_')
        logger.info(f'\t {t.capitalize():5s} {file.capitalize():5s} \t: {value}')

    with open(files['src_file'], encoding='utf-8') as src_file, \
            open(files['trg_file'], encoding='utf-8') as trg_file, \
            open(files['pred_file'], "a+", encoding='utf-8') as pred_file:

        for idx, (src_sent, trg_sent) in tqdm(enumerate(zip(src_file, trg_file))):
            # Strip white space and convert to lower
            src_sent = src_sent.strip().lower()
            trg_sent = trg_sent.strip().lower()

            # Tokenize
            src_tokens = [t.text for t in spacy_de.tokenizer(src_sent)]
            trg_tokens = [t.text for t in spacy_en.tokenizer(trg_sent)]

            # Translate with the trained model
            pred_tokens, attention = translate(src_tokens)
            pred_sent = " ".join(pred_tokens)

            # Save results to tensorboard
            tensorboard.add_text(f"{idx}/src", src_sent)
            tensorboard.add_text(f"{idx}/trg", trg_sent)
            tensorboard.add_text(f"{idx}/pred", pred_sent)
            tensorboard.add_scalar(f"{idx}/sent_bleu",
                                   sentence_bleu(trg_tokens, pred_tokens, smoothing_function=smoother))

            references.append(trg_tokens)
            hypotheses.append(pred_tokens)

            # Visualize the attention
            att_fig = display_attention(src_tokens, pred_tokens, attention)
            tensorboard.add_figure(f"{idx}/attention", att_fig)
            plt.close(att_fig)

            if idx == 10:
                break

            print(pred_sent, file=pred_file)

        tensorboard.add_scalar("test/corpus_blue", corpus_bleu(hypotheses, references, smoothing_function=smoother))


def get_translation_fn(model, device, SRC, TRG):
    def translate_fn(tokens):
        tokenized_sentence = ['<sos>'] + tokens + ['<eos>']
        numericalized = [SRC.vocab.stoi[t] for t in tokenized_sentence]
        sentence_length = torch.LongTensor([len(numericalized)]).to(device)
        tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
        translation_tensor_logits, attention = model(tensor, sentence_length, None, 0)
        translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
        translation = [TRG.vocab.itos[t] for t in translation_tensor]
        translation, attention = translation[1:], attention[1:]

        return translation, attention

    return translate_fn


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + sentence + ['<eos>'],
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
