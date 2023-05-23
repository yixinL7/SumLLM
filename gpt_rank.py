from gpt import gpt_critic_rank
import json
import random
import re
import argparse


def gpt_compare(src_dir, cand_dir_1, cand_dir_2, tgt_json_dir, tgt_txt_dir, model="gpt-3.5-turbo-0301"):
    with open(src_dir) as f:
        articles = [x.strip() for x in f.readlines()]

    with open(cand_dir_1) as f:
        summaries_1 = [x.strip() for x in f.readlines()]

    with open(cand_dir_2) as f:
        summaries_2 = [x.strip() for x in f.readlines()]

    with open("./prompts/gptcompare.txt") as f:
        prompt = f.read().strip()

    with open(tgt_json_dir, "w") as f1, open(tgt_txt_dir, "w") as f2:
        for i in range(len(articles)):
            print(i)
            article = articles[i]
            summary_1 = summaries_1[i]
            summary_2 = summaries_2[i]
            idxs = list(range(2))
            random.shuffle(idxs)
            summary_1, summary_2 = [summary_1, summary_2][idxs[0]], [summary_1, summary_2][idxs[1]]
            response, _ = gpt_critic_rank(article, [summary_1, summary_2], prompt, model)
            print(response["message"]["content"], file=f2)
            print(json.dumps({
                "article": article,
                "summary_1": summary_1,
                "summary_2": summary_2,
                "response": response,
                "idxs": idxs,
            }), file=f1)


def compute_scores(files, systems, output_dir=None):
    scores = {s: 0 for s in systems}
    data = []
    for fdir in files:
        with open(fdir) as f:
            data += [json.loads(line) for line in f.readlines()]
    output = dict()
    for (i, d) in enumerate(data):
        response = d["response"]["message"]["content"]
        # extract the decision with format: Decision: ...
        decision = re.findall(r"Decision: (.*)", response)[0]
        if "tie" not in decision and "Tie" not in decision:
            try:
                int(decision.strip())
                if int(decision.strip()) == 1:
                    scores[systems[d["idxs"][0]]] += 1
                    # print(systems[d["idxs"][0]])
                    output[i] = systems[d["idxs"][0]]
                elif int(decision.strip()) == 2:
                    scores[systems[d["idxs"][1]]] += 1
                    # print(systems[d["idxs"][1]])
                    output[i] = systems[d["idxs"][1]]
                else:
                    # print(decision)
                    pass
            except:
                if "Summary 1" in decision:
                    scores[systems[d["idxs"][0]]] += 1
                    # print(systems[d["idxs"][0]])
                    output[i] = systems[d["idxs"][0]]
                elif "Summary 2" in decision:
                    scores[systems[d["idxs"][1]]] += 1
                    # print(systems[d["idxs"][1]])
                    output[i] = systems[d["idxs"][1]]
                else:
                    # print(decision)
                    raise
        else:
            output[i] = "tie"
    print(scores)
    if output_dir is not None:
        with open(output_dir, "w") as f:
            json.dump(output, f, indent=4)
    return scores

def gpt_rank(article, summaries, model, tgt_dir):
    """
    article: str
    summaries: list of str
    model: str, OpenAI model name
    tgt_dir: str, target directory
    """
    with open("./prompts/gptrank.txt") as f:
        prompt = f.read().strip()
    response = gpt_critic_rank(article, summaries, prompt, model)
    with open(tgt_dir, "w") as f:
        print(json.dumps({
            "article": article,
            "summaries": summaries,
            "response": response,
        }), file=f)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-r", "--raw", action="store_true", help="get raw gpt score")
    parser.add_argument("-s", "--score", action="store_true", help="compute gpt score")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--cand_dir_1", type=str, help="candidate file 1")
    parser.add_argument("--cand_dir_2", type=str, help="candidate file 2")
    parser.add_argument("--tgt_json_dir", type=str, help="target json file")
    parser.add_argument("--tgt_txt_dir", type=str, help="target txt file")
    parser.add_argument("--model", type=str, help="model", default="gpt-3.5-turbo-0301")
    parser.add_argument("--system_1", type=str, help="system 1")
    parser.add_argument("--system_2", type=str, help="system 2")
    parser.add_argument("--output_dir", type=str, help="output dir")
    args = parser.parse_args()
    if args.raw:
        gpt_compare(
            args.src_dir,
            args.cand_dir_1,
            args.cand_dir_2,
            args.tgt_json_dir,
            args.tgt_txt_dir,
            args.model
        )
    elif args.score:
        compute_scores([args.tgt_json_dir], [args.system_1, args.system_2], args.output_dir)

