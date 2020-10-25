import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Constant as Constant
from tqdm import tqdm


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    trg = trg.transpose(0, 1)
    return trg, gold


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing):

    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction="sum")
    return loss


def train_epoch(model, training_data, optimizer, scheduler, device, smoothing=False):

    model.train()
    losses = []
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Training)   "
    for step, batch in enumerate(
        tqdm(training_data, mininterval=2, desc=desc, leave=False)
    ):
        batch_X, batch_Y = batch
        inp_data = patch_src(batch_X[0], Constant.PAD)
        target = patch_src(batch_Y[0], Constant.PAD)

        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss, n_correct, n_word = cal_performance(
            output, target, Constant.PAD, smoothing=False
        )
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, smoothing=False):

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Validation) "
    for step, batch in enumerate(
        tqdm(validation_data, mininterval=2, desc=desc, leave=False)
    ):
        batch_X, batch_Y = batch
        inp_data = patch_src(batch_X[0], Constant.PAD)
        target = patch_src(batch_Y[0], Constant.PAD)

        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        loss, n_correct, n_word = cal_performance(
            output, target, Constant.PAD, smoothing=False
        )
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total

    return loss_per_word, accuracy
