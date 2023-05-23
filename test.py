from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import argparse
from rouge_score import rouge_scorer


def generate_summaries_test(args):
    device = f"cuda:{args.gpuid}"
    mname = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(mname)
    max_length = 256
    min_length = 30
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    count = 1
    bsz = args.batch_size
    no_repeat_ngram_size = 3
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.force_bos_token_to_be_generated = False
    model.config.forced_bos_token_id = None
    with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % 10 == 0:
                print(count)
            if count % bsz == 0:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", padding="longest", truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        num_beams=num_beams,
                        max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        length_penalty=length_penalty,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    # print(dec)
                for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            if len(sline) == 0:
                sline = " "
            slines.append(sline.strip())
            count += 1
        if slines != []:
            with torch.no_grad():
                dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", padding="longest", truncation=True)
                summaries = model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    num_beams=num_beams,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                    early_stopping=True,
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            for hypothesis in dec:
                hypothesis = hypothesis.replace("\n", " ")
                fout.write(hypothesis + '\n')
                fout.flush()
        with open(args.ref_dir) as target, open(args.tgt_dir) as pred:
            cnt = 0
            rouge1, rouge2, rougeL = 0, 0, 0
            summary_lengths = []
            reference_lengths = []
            for tline, pline in zip(target, pred):
                tline = tline.strip()
                pline = pline.strip()
                scores = scorer.score(tline, pline)
                length = len(tokenizer.encode(pline))
                reference_lengths.append(len(tokenizer.encode(tline)))
                summary_lengths.append(length)
                rouge1 += scores['rouge1'].fmeasure
                rouge2 += scores['rouge2'].fmeasure
                rougeL += scores['rougeLsum'].fmeasure
                cnt += 1
            print(f"rouge1: {rouge1 / cnt}, rouge2: {rouge2 / cnt}, rougeL: {rougeL / cnt}")
            print(f"avg length: {sum(summary_lengths) / len(summary_lengths)}")
            print(f"avg reference length: {sum(reference_lengths) / len(reference_lengths)}")



if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, help="source file", default="cnndm.source.val.txt")
    parser.add_argument("--tgt_dir", type=str, help="target file", default="result")
    parser.add_argument("--ref_dir", type=str, help="reference file", default="cnndm.target.val.txt")
    parser.add_argument("--model_dir", type=str, help="model directory", default="model")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_beams", type=int, default=4, help="num beams")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="length penalty")
    args = parser.parse_args()
    generate_summaries_test(args)
   
