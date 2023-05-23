from torch.utils.data import Dataset
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class BrioDataset(Dataset):
    def __init__(self, fdir, model_type, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1, is_untok=True, is_pegasus=False, num=-1):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        with open(fdir) as f:
            self.data = [json.loads(x) for x in f]
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus
        if num > 0:
            self.data = self.data[:num]
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data = self.data[idx]
        src_txt = data["article"]
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        abstract = data["abstract"]
        if self.maxnum > 0:
            candidates = data["candidates"][:self.maxnum]
            data["candidates"] = candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            data["candidates"] = candidates
        # get the rank of the candidates
        scores = [x[1] for x in candidates]
        unique_scores = sorted(list(set(scores)))
        rank = [unique_scores.index(x) for x in scores]
        cand_txt = [abstract] + [x[0] for x in candidates]
        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand["input_ids"]
        if self.is_pegasus:
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids
        result = {
            "src_input_ids": src_input_ids, 
            "candidate_ids": candidate_ids,
            "rank": rank,
            }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp_brio(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    rank = torch.tensor([x["rank"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        "rank": rank,
        }
    if is_test:
        result["data"] = data
    return result



class BaseDataset(Dataset):
    def __init__(self, fdir, model_type, max_src_len=1024, max_tgt_len=512, is_test=False, is_pegasus=False):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        with open(fdir) as f:
            self.data = [json.loads(x) for x in f]
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.is_test = is_test
        self.is_pegasus = is_pegasus
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data = self.data[idx]
        src_txt = data["article"]
        src = self.tok([src_txt], max_length=self.max_src_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        abstract = data["abstract"]
        tgt = self.tok([abstract], max_length=self.max_tgt_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        tgt_input_ids = tgt["input_ids"]
        tgt_input_ids = tgt_input_ids.squeeze(0)
        if self.is_pegasus:
            # add start token
            _tgt_input_ids = tgt_input_ids.new_zeros(tgt_input_ids.size(0) + 1)
            _tgt_input_ids[1:] = tgt_input_ids.clone()
            _tgt_input_ids[0] = self.tok.pad_token_id
            tgt_input_ids = _tgt_input_ids
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp_base(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = pad([x["tgt_input_ids"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        }
    if is_test:
        result["data"] = data
    return result