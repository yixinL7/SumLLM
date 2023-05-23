import openai

def retry_with_backoff(func, max_retries=10, initial_wait_time=1):
    def wrapper(*args, **kwargs):
        retries = 0
        wait_time = initial_wait_time
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if retries == max_retries - 1:
                    raise e
                retries += 1
                wait_time *= 2
                print("retrying...", retries)
    return wrapper

openai.organization = "ORG_ID"
openai.api_key = "API_KEY"

@retry_with_backoff
def get_summary_score(article, summary, prompt, model="text-curie-001", temperature=0):
    prompt = prompt.replace("{{Article}}", article.replace("\n", " ").strip())
    prompt = prompt.replace("{{Summary}}", summary.replace("\n", " ").strip())
    response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=0, logprobs=1, echo=True)
    return response["choices"][0]


@retry_with_backoff
def gpt_critic_rank(article, summaries, prompt, model="gpt-3.5-turbo-0301"):
    prompt = prompt.replace("{{Article}}", article.replace("\n", " ").strip())
    # prompt = prompt.replace("{{Summary}}", summary.replace("\n", " ").strip())
    for i in range(len(summaries)):
        prompt = prompt.replace("{{Summary %d}}"%(i+1), summaries[i].replace("\n", " ").strip())
    # print(prompt)
    # quit()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=1024)
    # print(response)
    return response["choices"][0], prompt




