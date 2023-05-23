import json
from gpt import get_summary_score
import argparse

def _compute_gpt_score(tokens, logprobs, lp=1):
    # find the start token
    cnt = 0
    for i, x in enumerate(tokens):
        if x == "\n":
            cnt += 1
        if cnt == 4:
            assert tokens[i + 1] == "Summary"
            break
    start = i + 3
    logprobs = logprobs[start:]
    return sum(logprobs) / (len(logprobs) ** lp)

def compute_gpt_score(fpath, lp=1):
    with open(fpath) as f:
        data = [json.loads(x) for x in f]
    scores = [_compute_gpt_score(x["response"]["logprobs"]["tokens"], x["response"]["logprobs"]["token_logprobs"], lp=lp) for x in data]
    # print(sum(scores) / len(scores))
    print(f"{sum(scores) / len(scores):.3f}")

def get_raw_gpt_score(src_dir, tgt_dir, output_dir, model="text-davinci-003"):
    with open("./prompts/cnndm_length.txt") as f:
        prompt = f.read()
    with open(output_dir, "w") as f:
        with open(tgt_dir) as f2, open(src_dir) as f3:
            for i, (x, y) in enumerate(zip(f2, f3)):
                article = y.strip()
                article = article.replace("\n", " ")
                summary = x.strip()
                summary = summary.replace("\n", " ")
                response = get_summary_score(article, summary, prompt, model)
                print(json.dumps({"response": response, "id": i, "article": article, "summary": summary}), file=f)
                print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-r", "--raw", action="store_true", help="get raw gpt score")
    parser.add_argument("-s", "--score", action="store_true", help="compute gpt score")
    parser.add_argument("--src_dir", type=str, help="source file", default="cnndm.source.val.txt")
    parser.add_argument("--tgt_dir", type=str, help="target file", default="cnndm.target.val.txt")
    parser.add_argument("--output_dir", type=str, help="output file", default="cnndm.gpt.score.val.jsonl")
    parser.add_argument("--model", type=str, help="model", default="text-davinci-003")
    parser.add_argument("--length_penalty", type=int, default=1, help="length penalty")
    args = parser.parse_args()
    if args.raw:
        get_raw_gpt_score(args.src_dir, args.tgt_dir, args.output_dir, args.model)
    elif args.score:
        compute_gpt_score(args.output_dir, args.length_penalty)

    