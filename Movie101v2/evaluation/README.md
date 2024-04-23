# Evaluation

Evaluate the L1 and L2-Score of your generated narrations:

```python
from eval import *

client = OpenAI(api_key='your_openai_api_key')
language = 'zh' # 'zh' or 'en'
gt = 'This is a ground truth narration.'
pred = 'This is a generated narration.'

L1, L2 = eval_pred(client, language, gt, pred) # [[L1_env, L1_char], L2]
```