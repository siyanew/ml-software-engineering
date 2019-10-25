# %%
# Create helper functions
import torch

from pytorch.parse_config import ConfigParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(config):
    data_loader = config.init_obj_from_file('data_loader', test=True)
    model = config.init_obj_from_file('arch')
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    if 'packed' in config['arch'] and config['arch']['packed']:
        model.set_tokens(data_loader.SRC.vocab.stoi['<pad>'],
                         data_loader.TRG.vocab.stoi['<sos>'],
                         data_loader.TRG.vocab.stoi['<eos>'])

    model = model.to(device)
    model.eval()

    return (data_loader, model)


def load_sents(config):
    files = config['inference']
    with open(files['src_file'], encoding='utf-8') as src_file, \
            open(files['trg_file'], encoding='utf-8') as trg_file, \
            open(files['pred_file'], encoding='utf-8') as pred_file:
        src_lines = src_file.readlines()
        trg_lines = trg_file.readlines()
        pred_lines = pred_file.readlines()

    return (src_lines, trg_lines, pred_lines)


def analyse(model, idx, sents, data_loader):
    src_lines, trg_lines, pred_lines = sents
    src_sent = src_lines[idx]
    trg_sent = trg_lines[idx]

    translate = get_translation_fn(model, device, data_loader)

    src_sent = src_sent.strip().lower()
    trg_sent = trg_sent.strip().lower()

    # Tokenize
    src_tokens = data_loader.src_tokenize(src_sent)
    trg_tokens = data_loader.trg_tokenize(trg_sent)

    pred_tokens, attention = translate(src_tokens)
    pred_sent = " ".join(pred_tokens)

    print(f"DIFF: {src_sent} ")
    print(f"TRUE: {trg_sent}")
    print(f"PRED: {pred_sent}")

    att_fig = display_attention(src_tokens, pred_tokens, attention)
    att_fig.show()


# %%
# Load a model and corresponding data
import argparse

checkpoint = "saved/NMT1_preprocessed/model/model_best.pth"
config_p = "saved/NMT1_preprocessed/model/config.json"
args = argparse.ArgumentParser(description="analyse")
args.set_defaults(resume=checkpoint)
args.set_defaults(device=None)
args.set_defaults(config=config_p)
print(args)
config = ConfigParser.from_args(args)

data_loader, model = load_model(config)
sents = load_sents(config)

# %%
# Analyse a specific sentence

from test import display_attention, get_translation_fn

## Analysis for NMT1 model
# %% Attention works on ignore update '<file name>' pattern
idx = 86
analyse(model, idx, sents, data_loader)

# %% Attention does not work on any other sentence
idx = 145
analyse(model, idx, sents, data_loader)

# %% How many labels with ignore update '<file name>' pattern

trg_sents = sents[1]


def count_pattern(sent_set, s):
    count = 0
    for sent in sent_set:
        if s in sent.lower():
            count += 1

    print(f"Label with '{s}' pattern: {count} - {count / len(sent_set)}")
    return count


pattern = 'ignore update'
count_pattern(sents[1], pattern)
count_pattern(sents[2], pattern)

## Analysis for Java/C model
# %% Attention does not work on any other sentence
idx = 737
analyse(model, idx, sents, data_loader)

#%% analyse amount of update readme.md

pattern = 'update'
count_pattern(sents[1], pattern)
count_pattern(sents[2], pattern)

