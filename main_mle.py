import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from rouge_score import rouge_scorer
from transformers import BartTokenizer, PegasusTokenizer, BartForConditionalGeneration
from utils import Recorder
from data_utils import to_cuda, collate_mp_base, BaseDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model import label_smoothing_loss
from config import CONFIGS


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 100) 
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 4) # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 1000) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 5e-4) # max learning rate (* 1e-2)
    args.datatype = getattr(args, "datatype", "chatgpt_all") # data type
    args.dataset = getattr(args, "dataset", "cnndm") # dataset
    args.smooth = getattr(args, "smooth", 0.1) # label smoothing
    args.length_penalty = getattr(args, "length_penalty", 1.0) # length penalty
    args.do_sample = getattr(args, "do_sample", False) # whether to generate summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 256) # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 30) # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False) # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0) # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000) # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4) # number of beams for beam search
    args.max_src_len = getattr(args, "max_src_len", 1024) # max length of source article
    args.max_tgt_len = getattr(args, "max_tgt_len", 256) # max length of summary
    args.seq_avg = getattr(args, "seq_avg", False) # whether to use sequence-level average


def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    cnt = 0
    batch_cnt = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    length = 0
    device = f'cuda:{gpuid}' if args.cuda else "cpu"
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth, seq_avg=args.seq_avg)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    with torch.no_grad():
        # scoring
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, f"cuda:{gpuid}")
            text_id = batch["src_input_ids"]
            input_mask = text_id != tok.pad_token_id
            target_id = batch["tgt_input_ids"]
            target_mask = target_id != tok.pad_token_id
            target_mask[:, 0] = 1  # unmask the first token
            output = model(
                input_ids=text_id, 
                attention_mask=input_mask,
                decoder_input_ids=target_id, 
                decoder_attention_mask=target_mask,
                output_hidden_states=False
                )
            output = output[0]
            output = output[:, :-1]  # truncate last token
            gold = batch["tgt_input_ids"][:, 1:]  # shift right
            mle_loss += mle_fn(output.transpose(1, 2), gold)
            batch_cnt += 1

    mle_loss = mle_loss.item()

    if do_sample:
        if len(args.gpuid) > 1:
            _model = model.module
        else:
            _model = model
        with torch.no_grad():
            for (i, batch) in enumerate(gen_dataloader):
                if args.cuda:
                    to_cuda(batch, f"cuda:{gpuid}")
                text_id = batch["src_input_ids"]
                input_mask = text_id != tok.pad_token_id
                summaries = _model.generate(
                    input_ids=text_id,
                    attention_mask=input_mask,
                    max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (j, (hypothesis, d)) in enumerate(zip(dec, batch["data"])):
                    scores = scorer.score(d["abstract"], hypothesis)
                    rouge1 += scores["rouge1"].fmeasure
                    rouge2 += scores["rouge2"].fmeasure
                    rougeLsum += scores["rougeLsum"].fmeasure
                    length += len(tok.encode(hypothesis))
                    cnt += 1

    if len(args.gpuid) > 1:
        rouge1 = torch.FloatTensor([rouge1]).to(device)
        dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
        rouge1 = rouge1.item()
        rouge2 = torch.FloatTensor([rouge2]).to(device)
        dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
        rouge2 = rouge2.item()
        rougeLsum = torch.FloatTensor([rougeLsum]).to(device)
        dist.all_reduce(rougeLsum, op=dist.reduce_op.SUM)
        rougeLsum = rougeLsum.item()
        mle_loss = torch.FloatTensor([mle_loss]).to(device)
        dist.all_reduce(mle_loss, op=dist.reduce_op.SUM)
        mle_loss = mle_loss.item()
        cnt = torch.FloatTensor([cnt]).to(device)
        dist.all_reduce(cnt, op=dist.reduce_op.SUM)
        cnt = cnt.item()
        batch_cnt = torch.FloatTensor([batch_cnt]).to(device)
        dist.all_reduce(batch_cnt, op=dist.reduce_op.SUM)
        batch_cnt = batch_cnt.item()
        length = torch.FloatTensor([length]).to(device)
        dist.all_reduce(length, op=dist.reduce_op.SUM)
        length = length.item()

    if cnt > 0:
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        length = length / cnt
    mle_loss = mle_loss / batch_cnt   
    model.train()
    return {
        "mle_loss": mle_loss,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "length": length,
        }


def run(rank, args):
    if args.config is not None:
        CONFIGS[args.config](args)
    else:
        base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, args.log)
    # build dataloader
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_base, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_base, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = BaseDataset(f"./{args.dataset}/{args.datatype}/train.jsonl", args.model_type, args.max_src_len, args.max_tgt_len, False, args.is_pegasus)
    val_set = BaseDataset(f"./{args.dataset}/{args.datatype}/val.jsonl", args.model_type, args.max_src_len, args.max_tgt_len, True, args.is_pegasus)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.model_pt if args.model_pt is not None else args.model_type
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.force_bos_token_to_be_generated = False
    model.config.forced_bos_token_id = None
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()
    model.train()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth, seq_avg=args.seq_avg)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    optimizer = optim.Adam(model.parameters())
    if is_master:
        recorder.write_config(args, [model], __file__)
    minimum_mle_loss = 1e5
    minimum_generation_loss = 1e5
    all_step_cnt = 0
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # define evaluation function
    def eval_fn(rouge1, rouge2, rougeLsum):
        return 1 - (rouge1 * rouge2 + rougeLsum) / 3
    # start training
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        step_cnt = 0
        epoch_step = 0
        avg_mle_loss = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, f"cuda:{gpuid}")
            step_cnt += 1
            # forward pass
            text_id = batch["src_input_ids"]
            input_mask = text_id != tok.pad_token_id
            target_id = batch["tgt_input_ids"]
            target_mask = target_id != tok.pad_token_id
            target_mask[:, 0] = 1  # unmask the first token
            output = model(
                input_ids=text_id, 
                attention_mask=input_mask,
                decoder_input_ids=target_id, 
                decoder_attention_mask=target_mask,
                output_hidden_states=False
                )
            output = output[0]
            output = output[:, :-1]  # truncate last token
            gold = batch["tgt_input_ids"][:, 1:]  # shift right
            mle_loss = mle_fn(output.transpose(1, 2), gold)
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            loss = mle_loss
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                print("id: %d"%id)
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del loss, output

        result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
        # evaluate the model as a generator
        if args.do_sample:
            generation_loss = eval_fn(result["rouge1"], result["rouge2"], result["rougeLsum"])
        else:
            generation_loss = 0
        mle_loss = result["mle_loss"]
        if mle_loss < minimum_mle_loss and is_master:
            minimum_mle_loss = mle_loss
            if is_mp:
                recorder.save_pretrained(model.module, "model_mle")
            else:
                recorder.save_pretrained(model, "model_mle")
            recorder.print("best mle loss - epoch: %d"%(epoch))
        if generation_loss < minimum_generation_loss and is_master:
            minimum_generation_loss = generation_loss
            if is_mp:
                recorder.save_pretrained(model.module, "model_generation")
            else:
                recorder.save_pretrained(model, "model_generation")
            recorder.print("best generation loss - epoch: %d"%(epoch))
        if is_master:
            recorder.print("val generation loss: %.6f"%(generation_loss))
            recorder.print("val generation MLE loss: %.6f"%(result["mle_loss"]))
            recorder.print("val generation rouge1: %.6f"%(result["rouge1"]))
            recorder.print("val generation rouge2: %.6f"%(result["rouge2"]))
            recorder.print("val generation rougeLsum: %.6f"%(result["rougeLsum"]))
            recorder.print("val generation length: %.6f"%(result["length"]))
            if is_mp:
                recorder.save_pretrained(model.module, f"model_cur")
            else:
                recorder.save_pretrained(model, f"model_cur")


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default=None, type=str, help="model path")
    parser.add_argument("--config", default=None, type=str, help="config path")
    args = parser.parse_args()
    if args.cuda is False:
        main(args)
    else:
        if len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)