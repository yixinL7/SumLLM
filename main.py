import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from rouge_score import rouge_scorer
from transformers import BartTokenizer, PegasusTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp_brio, BrioDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model import BRIO, label_smoothing_loss, AdaptiveRankingLoss
from config import cnndm_setting, xsum_setting
import math
import json
import datetime


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 2) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 100) 
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 5) # accumulate gradients steps
    args.margin = getattr(args, "margin", math.log(2)) # margin for ranking loss on candidate summaries
    args.gold_margin = getattr(args, "gold_margin", 0) # margin for ranking loss on gold summaries
    args.gold_weight = getattr(args, "gold_weight", 0) # weight for ranking loss on gold summaries
    args.mle_weight = getattr(args, "mle_weight", 1) # weight for mle loss on gold summaries
    args.rank_weight = getattr(args, "rank_weight", 100) # weight for ranking loss on candidate summaries
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 400) # warmup steps
    args.normalize = getattr(args, "normalize", True) # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.no_gold = getattr(args, "no_gold", False) # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-4) # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 0.01) # scale of ranking loss
    args.score_mode = getattr(args, "score_mode", "log") # use log-likelihood for ranking loss
    args.datatype = getattr(args, "datatype", "brio_gpt4") # data type
    args.dataset = getattr(args, "dataset", "cnndm") # dataset
    args.max_len = getattr(args, "max_len", 256) # max length of summary
    args.max_num = getattr(args, "max_num", 8) # max number of candidate summaries
    args.smooth = getattr(args, "smooth", 0.1) # label smoothing
    args.total_len = getattr(args, "total_len", 1024) # total length of source article
    args.length_penalty = getattr(args, "length_penalty", 1.0) # length penalty
    args.do_sample = getattr(args, "do_sample", True) # whether to generate summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 256) # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 30) # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False) # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0) # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000) # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4) # number of beams for beam search


def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module.model
    else:
        _model = model.model
    cnt = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    ranking_loss = 0
    loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    _model.scoring_mode()
    with torch.no_grad():
        # scoring
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, device)
            samples = batch["data"]
            output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            _mle_loss = mle_fn(probs.transpose(1, 2), gold)
            _ranking_loss = AdaptiveRankingLoss(similarity, batch["rank"], args.margin, args.scale)
            mle_loss += _mle_loss
            ranking_loss += _ranking_loss
            # print(_mle_loss, _ranking_loss)
            loss += args.mle_weight * _mle_loss + args.rank_weight * _ranking_loss
            similarity = similarity.cpu().numpy()
            if i % 100 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = scorer.score(sample["abstract"], sents)
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / (i + 1)
    ranking_loss = ranking_loss / (i + 1)
    loss = loss / (i + 1)

    if len(args.gpuid) > 1:
        rouge1 = torch.FloatTensor([rouge1]).to(device)
        dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
        rouge1 = rouge1.item() / len(args.gpuid)
        rouge2 = torch.FloatTensor([rouge2]).to(device)
        dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
        rouge2 = rouge2.item() / len(args.gpuid)
        rougeLsum = torch.FloatTensor([rougeLsum]).to(device)
        dist.all_reduce(rougeLsum, op=dist.reduce_op.SUM)
        rougeLsum = rougeLsum.item() / len(args.gpuid)
        dist.all_reduce(mle_loss, op=dist.reduce_op.SUM)
        mle_loss = mle_loss / len(args.gpuid)
        dist.all_reduce(ranking_loss, op=dist.reduce_op.SUM)
        ranking_loss = ranking_loss / len(args.gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss / len(args.gpuid)

    mle_loss = mle_loss.item()
    ranking_loss = ranking_loss.item()
    loss = loss.item()
    
    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
    if do_sample:
        # generation
        _model.generation_mode()
        candidates, references = [], []
        with torch.no_grad():
            for (i, batch) in enumerate(gen_dataloader):
                if args.cuda:
                    to_cuda(batch, device)
                text_id = batch["src_input_ids"]
                input_mask = text_id != tok.pad_token_id
                summaries = _model.generate(
                    input_ids=text_id,
                    attention_mask=input_mask,
                    max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                samples = batch["data"]
                for (hypothesis, x) in zip(dec, samples):
                    hypothesis = hypothesis.replace("\n", " ")
                    candidates.append(hypothesis)
                    ref = x["abstract"]
                    references.append(ref)
                    # print(ref)
                    score = scorer.score(ref, hypothesis)
                    sample_rouge1 += score["rouge1"].fmeasure
                    sample_rouge2 += score["rouge2"].fmeasure
                    sample_rougeLsum += score["rougeLsum"].fmeasure
                    cnt += 1
        with open(os.path.join(args.dir, f"{args.epoch}_{gpuid}.jsonl"), "w") as f:
            for (hypothesis, reference) in zip(candidates, references):
                print(json.dumps({"hypothesis": hypothesis, "reference": reference}), file=f)
        _model.scoring_mode()
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougeLsum,
        "mle_loss": mle_loss,
        "ranking_loss": ranking_loss,
        "loss": loss,
        } 


def run(rank, args):
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
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
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = BrioDataset(f"./{args.dataset}/{args.datatype}/train.jsonl", args.model_type, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    val_set = BrioDataset(f"./{args.dataset}/{args.datatype}/val.jsonl", args.model_type, is_test=True, max_len=256, is_sorted=True, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=4*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=4*args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.model_pt if args.model_pt is not None else args.model_type
    model = BRIO(model_path)
    model.model.config.decoder_start_token_id = tok.bos_token_id
    model.model.config.force_bos_token_to_be_generated = False
    model.model.config.forced_bos_token_id = None
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()
    model.train()
    # set the model to scoring mode
    if is_mp:
        model.module.model.scoring_mode()
    else:
        model.model.scoring_mode()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    optimizer = optim.Adam(model.parameters())
    if is_master:
        recorder.write_config(args, [model], __file__)
    minimum_ranking_loss = 100
    minimum_mle_loss = 1e5
    all_step_cnt = 0
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
        date = datetime.datetime.now().strftime("%y-%m-%d")
        args.dir = f"./cache/{date}-{id}"
    else:
        date = datetime.datetime.now().strftime("%y-%m-%d")
        args.dir = f"./cache/{date}-{id}"
    # define evaluation function
    def eval_fn(rouge1, rouge2, rougeLsum):
        return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    # start training
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        avg_ranking_loss = 0
        avg_mle_loss = 0
        avg_gold_margin_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            # forward pass
            output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            gold_margin_loss = torch.relu(similarity - gold_similarity[:, None]).mean()
            ranking_loss = AdaptiveRankingLoss(similarity, batch["rank"], args.margin, args.scale)
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            mle_loss = mle_fn(probs.transpose(1, 2), gold)
            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss + args.gold_weight * gold_margin_loss
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            avg_gold_margin_loss += gold_margin_loss.item() / args.accumulate_step
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
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f, avg gold margin loss: %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq, avg_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq, avg_gold_margin_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.plot("mle_loss", {"loss": avg_mle_loss / args.report_freq}, all_step_cnt)
                recorder.plot("ranking_loss", {"loss": avg_ranking_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_mle_loss, avg_ranking_loss, avg_loss, avg_gold_margin_loss = 0, 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

        
        # evaluate the model as a scorer
        args.epoch = epoch
        result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
        loss = result["loss"]
        if loss < minimum_ranking_loss and is_master:
            minimum_ranking_loss = loss
            if is_mp:
                recorder.save_pretrained(model.module, "model_ranking")
            else:
                recorder.save_pretrained(model, "model_ranking")
            recorder.print("best ranking loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
        if is_master:
            recorder.print("val loss: %.6f"%(loss))
            recorder.print("val ranking loss: %.6f"%(result["ranking_loss"]))
            recorder.print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
            %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
        # evaluate the model as a generator
        if args.do_sample:
            mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
        else:
            mle_loss = result["mle_loss"]
        if mle_loss < minimum_mle_loss and is_master:
            minimum_mle_loss = mle_loss
            if is_mp:
                recorder.save_pretrained(model.module, "model_generation")
            else:
                recorder.save_pretrained(model, "model_generation")
            recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
        if is_master:
            recorder.print("val generation loss: %.6f"%(mle_loss))
            if args.do_sample:
                recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
        # save current model
        if is_master:
            recorder.save(optimizer, "optimizer.bin")


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
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default=None, type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    args = parser.parse_args()
    if args.cuda is False:
        main(args)
    else:
        if len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)