from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset, Dataset
from pairgen.generation import generate_question_answer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model.to("cuda")

ds = load_dataset("kardosdrur/dawiki_title_content", split="all")
ds = ds.shuffle()

def generate_dataset(passages, model, tokenizer):
    for passage in passages:
        try:
            question, answer = generate_question_answer(passage, model, tokenizer, device="cuda")
            if bool(question) and bool(answer):
                yield dict(question=question, answer=answer)
        except Exception as e:
            print("Exception happened, skipping.")
            print(e)
            continue


passages = [row["content"] for row in ds]
passages = tqdm(passages, desc="Generating Questions and Answers with Zephyr")

entries = []
for entry in generate_dataset(passages, model, tokenizer):
    entries.append(entry)
    qa_ds = Dataset.from_list(entries)
    qa_ds.save_to_disk("dawiki_qa")
