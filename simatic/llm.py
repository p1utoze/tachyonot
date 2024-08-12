from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


class SimaticLLM:
    def __init__(self, checkpoint="Qwen/Qwen1.5-0.5B-Chat", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def generate(self, messages, max_new_tokens, device, temperature, top_p, do_sample, stream=False):
        if stream:
            return self._stream_generate(messages, max_new_tokens, temperature, top_p, do_sample)
        else:
            return self._generate(messages, max_new_tokens, temperature, top_p, do_sample)

    def _generate(self, messages, max_new_tokens, temperature, top_p, do_sample, device="cpu"):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            **model_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _stream_generate(self, messages, max_new_tokens, temperature, top_p, do_sample, device="cpu"):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generation_kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            streamer=streamer,
            ** model_inputs,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer


if __name__ == "__main__":
    simatic_llm = SimaticLLM()
    messages = [{"role": "user", "content": "What is Augmented reality?"}]
    response = simatic_llm.generate(messages, max_new_tokens=256, device="cpu", temperature=0.1, top_p=0.92,
                                    do_sample=True)
    print("Response:", response)