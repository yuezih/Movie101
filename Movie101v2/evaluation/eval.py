
import time
from openai import OpenAI
import time


def call_openai(client, prompt):
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
                model = "gpt-3.5-turbo",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            print("Continue......")
            time.sleep(10)


def load_template_prompt(level, language):
    with open(f'./prompt/{level}_{language}.txt', 'r') as f:
        return f.read()


def build_prompt(prompt, gt, pred):
    prompt = prompt.replace('<gt>', gt)
    prompt = prompt.replace('<pred>', pred)
    return prompt


def scoring_L1(client, language, gt, pred):
    prompt = build_prompt(load_template_prompt('L1', language), gt, pred)
    response = call_openai(client, prompt)
    while len(response) > 4:
        print('Wrong format answer. Retrying L1 scoring.')
        response = call_openai(client, prompt)
    try:
        score = [int(x) for x in response.split(',')]
        assert len(score) == 2
        assert 0 <= score[0] <= 5 and 0 <= score[1] <= 5
    except:
        print(f'Error: {response}')
        score = [-1, -1]

    return score


def scoring_L2(client, language, gt, pred):
    prompt = build_prompt(load_template_prompt('L2', language), gt, pred)
    response = call_openai(client, prompt)
    while len(response.strip()) != 1:
        print('Wrong format answer. Retrying L2 scoring.')
        response = call_openai(client, prompt)
    try:
        score = int(response.strip())
        assert 0 <= score <= 5
    except:
        print(f'Error: {response}')
        score = -1

    return score


def eval_pred(client, language, gt, pred):
    L1 = scoring_L1(client, language, gt, pred)
    L2 = scoring_L2(client, language, gt, pred)
    return [L1, L2]


if __name__ == '__main__':
    lang = input("Input language (zh/en):")
    pred = input("Input prediction:")
    gt = input("Input ground truth:")
    client = OpenAI(api_key='') # Add your OpenAI API key here
    print(eval_pred(client, lang, gt, pred))