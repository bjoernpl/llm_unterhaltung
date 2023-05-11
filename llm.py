import openai
from loguru import logger
try:
    from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
except ImportError:
    logger.warning("Could not import transformers, some functionality might be missing.")
    pipeline = None
    AutoTokenizer = None
    TextIteratorStreamer = None
from threading import Thread

base_prompt = "Du bist ein hilfsbereiter Chatbot namens Thorsten."

class StreamingGPTProcessor:
    """A class that wraps the OpenAI API for GPT-3 or 4 and provides a simple interface for chatbots.
    Explicitly uses streaming output, return values are generators."""
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        openai.api_key_path = "api_key.txt"

    def _call_chat(self, prompt, user_history, bot_history):
        messages = [{"role": "system", "content": base_prompt}]
        i = 0
        while i < len(user_history):
            messages += [{"role": "user", "content": user_history[i]}]
            if len(bot_history) > i:
                messages += [{"role": "assistant", "content": bot_history[i]}]
            i += 1
        messages += [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        full_text = ""
        for i, chunk in enumerate(response):
            if i == 0:
                continue
            a = chunk['choices'][0]['delta']
            if "content" not in a:
                return
            full_text += a["content"]
            yield full_text
    
    def _call_single(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": prompt},
                ],
            stream=True
        )
        full_text = ""
        for i, chunk in enumerate(response):
            if i == 0:
                continue
            a = chunk['choices'][0]['delta']
            if "content" not in a:
                return
            full_text += a["content"]
            yield full_text

    def __call__(self, prompt, user_history=None, bot_history=None):
        if user_history is None:
            return self._call_single(prompt)
        else:
            return self._call_chat(prompt, user_history, bot_history)

class GPTProcessor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        openai.api_key_path = "api_key.txt"

    def _call_chat(self, prompt, user_history, bot_history):
        messages = [{"role": "system", "content": base_prompt}]
        i = 0
        while i < len(user_history):
            messages += [{"role": "user", "content": user_history[i]}]
            if len(bot_history) > i:
                messages += [{"role": "assistant", "content": bot_history[i]}]
            i += 1
        messages += [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        message = response['choices'][0]['message']['content']
        return message
    
    def _call_single(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": prompt},
                ]
        )
        message = response['choices'][0]['message']['content']
        return message
    
    def __call__(self, prompt, user_history=None, bot_history=None):
        if user_history is None:
            return self._call_single(prompt)
        else:
            return self._call_chat(prompt, user_history, bot_history)


class LLMProcessor:
    def __init__(self, model_name, stream=False):
        self.stream = stream
        self.model_name = model_name
        logger.info(f"Loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.generate = pipeline(model=self.model_name, trust_remote_code=True)
        logger.info(f"Loaded model {model_name}")

    def _call_stream(self, prompt):
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        t = Thread(target=self.generate, args=[prompt], kwargs={"streamer": streamer})
        t.start()
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield partial_text

    def _call(self, prompt):
        return self.generate(prompt)
    
    def __call__(self, prompt):
        if self.stream:
            return self._call_stream(prompt)
        else:
            return self._call(prompt)
        
class LLMChatProcessor:
    """Todo: Implement chat processor"""
    pass

