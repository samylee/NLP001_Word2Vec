import torch
from torch.utils.data import DataLoader
from functools import partial
from torchtext.data.utils import get_tokenizer
from torchtext.data import to_map_style_dataset
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator


def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer('basic_english', language='en')
    return tokenizer


def get_data_iterator(data_dir, dataset_type):
    data_iter = WikiText2(root=data_dir, split=(dataset_type))
    data_iter = to_map_style_dataset(data_iter)
    return data_iter


def build_vocab(data_iter, tokenizer, min_freq=50):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=['<unk>'],
        min_freq=min_freq,
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def collate_cbow(batch, text_pipeline, cbow_n_words=4, max_len=256):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < cbow_n_words * 2 + 1:
            continue

        if max_len:
            text_tokens_ids = text_tokens_ids[:max_len]

        for idx in range(len(text_tokens_ids) - cbow_n_words * 2):
            token_id_sequence = text_tokens_ids[idx: (idx + cbow_n_words * 2 + 1)]
            output = token_id_sequence.pop(cbow_n_words)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_data_loader(data_dir, train_bs, val_bs):
    # get tokenizer
    tokenizer = get_english_tokenizer()

    # train dataloader
    train_dataiter = get_data_iterator(data_dir, dataset_type='train')
    # get vocab
    vocab = build_vocab(train_dataiter, tokenizer)
    train_dataloader = DataLoader(
        train_dataiter,
        batch_size=train_bs,
        shuffle=True,
        collate_fn=partial(collate_cbow, text_pipeline=lambda x: vocab(tokenizer(x))),
    )

    # val dataloader
    val_dataiter = get_data_iterator(data_dir, dataset_type='valid')
    val_dataloader = DataLoader(
        val_dataiter,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=partial(collate_cbow, text_pipeline=lambda x: vocab(tokenizer(x))),
    )

    return train_dataloader, val_dataloader, len(vocab.get_stoi())