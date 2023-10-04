
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


# build claude class
class Claude_Model():
    def __init__(self, model="claude-2", api_key="", temperature=0, max_tokens=1024, n=1, patience=1000, sleep_time=0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time

    def get_response(self, image_path, user_prompt):
        patience = self.patience
        while patience > 0:
            patience -= 1
            try:
                # configure the default for all requests:
                anthropic = Anthropic(
                    max_retries=0,
                    api_key=self.api_key,
                )

                # update prompt
                if "python" in user_prompt:
                    _HUMAN_PROMPT = HUMAN_PROMPT + "Generate the runnable python code only."
                else:
                    _HUMAN_PROMPT = HUMAN_PROMPT

                # configure per-request options
                completion = anthropic.with_options(max_retries=5).completions.create(
                    prompt=f"{_HUMAN_PROMPT} {user_prompt}{AI_PROMPT}",
                    max_tokens_to_sample=self.max_tokens,
                    model=self.model,
                )
                
                # inference
                prediction = completion.completion.strip()
                if "python" in user_prompt:
                    prediction = prediction.replace("```python", "").replace("```", "").strip()
                if prediction != "" and prediction != None:
                    return prediction
                        
            except Exception as e:
                if "limit" not in str(e):
                    print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""
