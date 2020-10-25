import spacy
from torchtext.data.metrics import bleu_score
import sys
from sklearn.utils import shuffle
from nltk import bleu_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Constant as Constant
from tqdm import tqdm
from model import Transformer
import math
import time


random_state = 42


def translate_sentence(model, sentence_ids, sp, device, max_length=50):
    sentence_tensor = torch.LongTensor(sentence_ids).unsqueeze(1).to(device)
    outputs = [Constant.BOS]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        # TODO beam searchの実装
        #  outputの二次元目がビーム幅

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == Constant.EOS:
            break
    return outputs


def bleu_fromdatalist(data_X, data_Y, model, sp, device):
    targets = []
    outputs = []

    for src, trg in tqdm(zip(data_X, data_Y)):
        prediction = translate_sentence(model, src, sp, device)
        source = sp.decode(prediction).split()
        target = sp.decode(trg).split()
        outputs.append(source)
        targets.append([target])

    return bleu_score.corpus_bleu(targets, outputs)


def load_model(checkpoint, device):
    model_args = checkpoint["settings"]

    model = Transformer(
        model_args["embedding_size"],
        model_args["src_vocab_size"],
        model_args["tgt_vocab_size"],
        model_args["src_pad_idx"],
        model_args["num_heads"],
        model_args["num_encoder_layers"],
        model_args["num_decoder_layers"],
        model_args["forward_expansion"],
        model_args["dropout"],
        model_args["max_len"],
        model_args["device"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    print("[Info] Trained model state loaded.")
    return model


def load_data(file_path_src, file_path_trg, sp):
    datas = []
    datat = []
    liness = open(file_path_src, encoding="utf-8")
    linest = open(file_path_trg, encoding="utf-8")
    for lines, linet in tqdm(zip(liness, linest)):
        encoded_ids_s = sp.encode(lines)
        encoded_ids_t = sp.encode(linet)
        datas.append([Constant.BOS] + encoded_ids_s + [Constant.EOS])
        datat.append([Constant.BOS] + encoded_ids_t + [Constant.EOS])
    return datas, datat


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in tqdm(data):
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def print_performances(header, loss, accu, start_time):
    print(
        "  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, "
        "elapse: {elapse:3.3f} min".format(
            header=f"({header})",
            ppl=math.exp(min(loss, 100)),
            accu=100 * accu,
            elapse=(time.time() - start_time) / 60,
        )
    )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


class DataLoader(object):
    def __init__(self, src_insts, tgt_insts, batch_size, device, shuffle=True):
        """
        :param src_insts: list, 入力言語の文章（単語IDのリスト）のリスト
        :param tgt_insts: list, 出力言語の文章（単語IDのリスト）のリスト
        :param batch_size: int, バッチサイズ
        :param shuffle: bool, サンプルの順番をシャッフルするか否か
        """
        self.data = list(zip(src_insts, tgt_insts))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0
        self.device = device

        self.reset()

    def reset(self):
        if self.shuffle:
            self.data = shuffle(self.data, random_state=random_state)
        self.start_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        def preprocess_seqs(seqs):
            # パディング
            max_length = max([len(s) for s in seqs])
            data = [s + [Constant.PAD] * (max_length - len(s)) for s in seqs]
            # 単語の位置を表現するベクトルを作成
            positions = [
                [pos + 1 if w != Constant.PAD else 0 for pos, w in enumerate(seq)]
                for seq in data
            ]
            # テンソルに変換
            data_tensor = torch.tensor(data, dtype=torch.long, device=self.device)
            position_tensor = torch.tensor(
                positions, dtype=torch.long, device=self.device
            )
            return data_tensor, position_tensor

        # ポインタが最後まで到達したら初期化する
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        # バッチを取得して前処理
        src_seqs, tgt_seqs = zip(
            *self.data[self.start_index : self.start_index + self.batch_size]
        )
        src_data, src_pos = preprocess_seqs(src_seqs)
        tgt_data, tgt_pos = preprocess_seqs(tgt_seqs)

        # ポインタを更新する
        self.start_index += self.batch_size

        return (src_data, src_pos), (tgt_data, tgt_pos)


# def translate_sentence(model, sentence, german, english, device, max_length=50):
#     # Load german tokenizer
#     spacy_ger = spacy.load("de")

#     # Create tokens using spacy and everything in lower case (which is what our vocab is)
#     if type(sentence) == str:
#         tokens = [token.text.lower() for token in spacy_ger(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]

#     # Add <SOS> and <EOS> in beginning and end respectively
#     tokens.insert(0, german.init_token)
#     tokens.append(german.eos_token)

#     # Go through each german token and convert to an index
#     text_to_indices = [german.vocab.stoi[token] for token in tokens]

#     # Convert to Tensor
#     sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

#     outputs = [english.vocab.stoi["<sos>"]]
#     for i in range(max_length):
#         trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

#         with torch.no_grad():
#             output = model(sentence_tensor, trg_tensor)

#         best_guess = output.argmax(2)[-1, :].item()
#         outputs.append(best_guess)

#         if best_guess == english.vocab.stoi["<eos>"]:
#             break

#     translated_sentence = [english.vocab.itos[idx] for idx in outputs]
#     # remove start token
#     return translated_sentence[1:]
