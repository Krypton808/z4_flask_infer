from transformers import AutoTokenizer
import time
try:
    import config
except:
    from gradio_infer.base.tgi_infer import config
import uuid


class Processor(object):
    def __init__(
            self,
            model_path_or_name: str = "/mnt/nfs/yechen/models/tigerbot-13b-2h-sft-20g-mix0.0-group"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_name,
            cache_dir=None,
            padding_side="left",
            truncation_side='left',
            padding=True,
            truncation=True,
            add_bos_token=False,
        )
        self.tok_ins = "\n\n### Instruction:\n"
        self.tok_res = "\n\n### Response:\n"

    def preprocess(
            self,
            messages: list = None,
            do_sample: bool = None,
            top_p: float = None,
            top_k: int = None,
            temperature: float = None,
            max_input_length: int = None,
            max_output_length: int = None,
            repetition_penalty: float = None
    ):
        if do_sample is None:
            do_sample = config.MODEL_PARAMETER_DEFAULT_DO_SAMPLE
        if top_p is None:
            top_p = config.MODEL_PARAMETER_DEFAULT_TOP_P
        if top_k is None:
            top_k = config.MODEL_PARAMETER_DEFAULT_TOP_K
        if temperature is None:
            temperature = config.MODEL_PARAMETER_DEFAULT_TEMPERATURE
        if max_input_length is None:
            max_input_length = config.MODEL_PARAMETER_DEFAULT_MAX_INPUT_LENGTH
        if max_output_length is None:
            max_output_length = config.MODEL_PARAMETER_DEFAULT_MAX_OUTPUT_LENGTH
        if repetition_penalty is None:
            repetition_penalty = config.MODEL_PARAMETER_DEFAULT_REPETITION_PENALTY

        if top_p is not None and (top_p <= 0.0 or top_p > 1):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if temperature is not None and temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}.")
        if max_input_length is not None and max_input_length < 1:
            raise ValueError(f"max_input_length must be at least 1, got {max_input_length}.")
        if max_output_length is not None and max_output_length < 1:
            raise ValueError(f"max_output_length must be at least 1, got {max_output_length}.")

        input_text = ""
        for message in messages:
            if "role" not in message or 'content' not in message:
                continue
            if message['role'] == 'system':
                input_text += self.tok_ins + message['content']
            elif message['role'] == 'user':
                input_text += self.tok_ins + message['content']
            elif message['role'] == 'assistant':
                input_text += self.tok_res + message['content'] + "</s>"

        if not input_text.endswith(self.tok_res):
            input_text += self.tok_res

        inputs = self.tokenizer(input_text, truncation=True, max_length=max_input_length)
        text = self.tokenizer.decode(inputs["input_ids"])
        text = "\n".join([t.strip() for t in text.split("\n")])

        generation_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "max_new_tokens": max_output_length,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
        }
        return {"inputs": text, "parameters": generation_kwargs}

    def postprocess(self, text, generated_text):
        input_token_length = len(self.tokenizer(text)["input_ids"])
        answer_token_length = len(self.tokenizer(generated_text)["input_ids"])

        result = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.MODEL,
            "system_fingerprint": "v1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_token_length,
                "completion_tokens": answer_token_length,
                "total_tokens": input_token_length + answer_token_length
            }
        }
        return result


if __name__ == "__main__":
    processor = Processor()
    inputs = processor.preprocess(query="那英国的呢", session=[{"human": "法国的首度在哪里", "assistant": "巴黎"}])
    print(inputs)
    from text_generation import Client

    client = Client("http://127.0.0.1:8080", timeout=60)
    output = client.generate(inputs["inputs"], **inputs["parameters"])
    text = inputs["inputs"]
    generated_text = output.generated_text
    print(processor.postprocess(text=text, generated_text=generated_text))
