from tqdm import tqdm

if __name__ == '__main__':
    DIFF_TOKENS = 100
    MSG_TOKENS = 30

    src_path = 'data/Top1000-Java/java-top-1000.processed.diff'
    trg_path = 'data/Top1000-Java/java-top-1000.processed.msg'
    src_path_short = 'data/Top1000-Java/java-top-1000.processed.100.diff'
    trg_path_short = 'data/Top1000-Java/java-top-1000.processed.30.diff'

    src_file = open(src_path, encoding='utf-8')
    trg_file = open(trg_path, encoding='utf-8')

    src_short = open(src_path, encoding='utf-8')
    trg_short = open(src_path, encoding='utf-8')

    for idx, (src_sent, trg_sent) in tqdm(enumerate(zip(src_file, src_path))):
        diff_tokens = src_sent.strip().split(" ")
        msg_tokens = trg_sent.strip().split(" ")

        # Filter on the number of tokens in the diff
        if len(diff_tokens) >= DIFF_TOKENS:
            continue

        # Filter on the number of tokens in the msg
        if len(msg_tokens) >= MSG_TOKENS:
            continue

        print(" ".join(diff_tokens), file=src_short)
        print(" ".join(msg_tokens), file=trg_short)

    src_file.close()
    trg_file.close()
    src_short.close()
    trg_short.close()
