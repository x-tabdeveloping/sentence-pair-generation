from typing import Tuple

question_answer_prompt = """
<|system|>
You are a question asker.
Whatever the user gives you, you ask a question
that is related to it, and provide an answer.
You may only respond in Danish. Prefix your questions with Spørgsmål: and answers with Svar:
</s>
<|user|>
{paragraph}
</s>
<|assistant|>
"""
def generate_question_answer(paragraph: str, model, tokenizer, device="cpu") -> Tuple[str, str]:
    prompt = question_answer_prompt.format(paragraph=paragraph)
    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_input, max_new_tokens=256)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("<|assistant|>")[1]
    response = response.strip().removeprefix("Spørgsmål:")
    try:
        question, answer, *_ = response.split("Svar:")
        return question.strip(), answer.strip()
    except ValueError:
        return "", ""


title_prompt = """
<|system|>
You are a Danish title giver.
When the user gives you a paragraph you should respond with a short title for it.
You may only respond in Danish, no English is allowed.
Respond with a title only, do not repeat the paragraph.
Prefix your responses with Titel:
</s>
<|user|>
{paragraph}
</s>
<|assistant|>
"""
def generate_title(paragraph: str, model, tokenizer, device="cpu") -> str:
    prompt = title_prompt.format(paragraph=paragraph)
    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_input, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("<|assistant|>")[1]
    response = response.strip()
    if "\n" in response:
        response = response.split("\n")[0]
    return response.removeprefix("Titel:").strip()


title_prompt = """
<|system|>
You are a Danish paraphraser.
When the user gives you a paragraph you should respond with a paraphrased version of it,
keeping as little of the original paragraph as possible.
You may only respond in Danish, no English is allowed.
Prefix your responses with Omformulering:
</s>
<|user|>
{paragraph}
</s>
<|assistant|>
"""
def generate_paraphrase(paragraph: str, model, tokenizer, device="cpu") -> str:
    prompt = title_prompt.format(paragraph=paragraph)
    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_input, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("<|assistant|>")[1]
    response = response.strip()
    if "\n" in response:
        response = response.split("\n")[0]
    return response.removeprefix("Omformulering:").strip()

