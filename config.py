import math


def chatgpt_warmup(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 4) # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 1000) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 5e-4) # max learning rate (* 1e-2)
    args.datatype = getattr(args, "datatype", "chatgpt") # data type
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


def gpt3_mle(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 4) # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 100) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-4) # max learning rate (* 1e-2)
    args.datatype = getattr(args, "datatype", "gpt3_all") # data type
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


def gpt3_mle_small(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 4) # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 100) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-4) # max learning rate (* 1e-2)
    args.datatype = getattr(args, "datatype", "gpt3") # data type
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


def gpt3_brio(args):
    args.batch_size = getattr(args, 'batch_size', 2) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 5) # accumulate gradients steps
    args.margin = getattr(args, "margin", math.log(2)) # margin for ranking loss on candidate summaries
    args.gold_margin = getattr(args, "gold_margin", 0) # margin for ranking loss on gold summaries
    args.gold_weight = getattr(args, "gold_weight", 0) # weight for ranking loss on gold summaries
    args.mle_weight = getattr(args, "mle_weight", 1) # weight for mle loss on gold summaries
    args.rank_weight = getattr(args, "rank_weight", 100) # weight for ranking loss on candidate summaries
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 100) # warmup steps
    args.normalize = getattr(args, "normalize", True) # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.no_gold = getattr(args, "no_gold", False) # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 1e-4) # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 0.01) # scale of ranking loss
    args.score_mode = getattr(args, "score_mode", "log") # use log-likelihood for ranking loss
    args.datatype = getattr(args, "datatype", "gpt_brio_gptscore") # data type
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


def chatgpt_mle(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
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


def chatgpt_brio(args):
    args.batch_size = getattr(args, 'batch_size', 2) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
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
    args.datatype = getattr(args, "datatype", "brio_chatgpt") # data type
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


def gpt4_mle(args):
    args.batch_size = getattr(args, 'batch_size', 8) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 4) # accumulate gradients steps
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 100) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-4) # max learning rate (* 1e-2)
    args.datatype = getattr(args, "datatype", "gpt4") # data type
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


def chatgpt_brio(args):
    args.batch_size = getattr(args, 'batch_size', 2) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 10) 
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


CONFIGS = {
    "chatgpt_warmup": chatgpt_warmup,
    "gpt3_mle": gpt3_mle,
    "gpt3_mle_small": gpt3_mle_small,
    "gpt3_brio": gpt3_brio,
    "chatgpt_mle": chatgpt_mle,
    "chatgpt_brio": chatgpt_brio,
    "gpt4_mle": gpt4_mle,
    "chatgpt_brio": chatgpt_brio,
}