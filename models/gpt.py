import time
import openai


# build gpt class
class GPT_Model():
    def __init__(self, model="gpt-3.5-turbo", api_key="", temperature=0, max_tokens=1024, n=1, patience=1000000, sleep_time=0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.patience = patience
        self.sleep_time = sleep_time

    def get_response(self, image_path, user_prompt):
        patience = self.patience
        max_tokens = self.max_tokens
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        while patience > 0:
            patience -= 1
            try:
                # print("self.model", self.model)
                response = openai.ChatCompletion.create(model=self.model,
                                                        messages=messages,
                                                        api_key=self.api_key,
                                                        temperature=self.temperature,
                                                        max_tokens=max_tokens,
                                                        n=self.n
                                                        )
                if self.n == 1:
                    prediction = response['choices'][0]['message']['content'].strip()
                    if prediction != "" and prediction != None:
                        return prediction
                else:
                    prediction = [choice['message']['content'].strip() for choice in response['choices']]
                    if prediction[0] != "" and prediction[0] != None:
                        return prediction
                        
            except Exception as e:
                if "limit" not in str(e):
                    print(e)
                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!!Reduce max_tokens to", max_tokens)
                if max_tokens < 8:
                    return ""
                if "Please reduce the length of the messages." in str(e):
                    print("!!Reduce user_prompt to", user_prompt[:-1])
                    return ""
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""
