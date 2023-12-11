from pathlib import Path
from typing import List, Dict, Set
import json
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange, tqdm
from datasets import load_dataset, Dataset
from pairgen.generation import generate_question_answer

NUM_SHARDS = 500


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as in_file:
        for line in in_file:
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(path: str, records: List[Dict]):
    records_str = [json.dumps(record) for record in records]
    with open(path, "w") as out_file:
        out_file.write("\n".join(records_str))

def load_shards(shards: List[str]) -> List[Dict]:
    records = []
    for shard in shards:
        records.extend(load_jsonl(shard))
    return records

def get_shard_ids(shards: List[str]) -> Set[int]:
    ids = []
    for shard in shards:
        shard = Path(shard)
        _, shard_id = shard.stem.split("_")
        shard_id = int(shard_id)
        ids.append(shard_id)
    return set(ids)

def generate_dataset(passages, model, tokenizer):
    for passage in passages:
        try:
            question, answer = generate_question_answer(
                passage, model, tokenizer, device="cuda"
            )
            if bool(question) and bool(answer):
                yield dict(question=question, answer=answer)
        except Exception as e:
            print("Exception happened, skipping.")
            print(e)
            continue

def push_all_shards(shards_folder: Path, repo: str):
    shards = glob(str(shards_folder.joinpath("*.jsonl")))
    data = load_shards(shards)
    ds = Dataset.from_list(data)
    ds.push_to_hub(repo)


print("Loading Zephyr from repositories.")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model.to("cuda")

print("Loading Wiki")
ds = load_dataset("kardosdrur/dawiki_title_content", split="all")
ds = ds.shuffle(seed=42)

print("Collecting all available shards.")
shards_folder = Path("dawiki_qa/shards/")
shards_folder.mkdir(exist_ok=True, parents=True)
shards = glob(str(shards_folder.joinpath("*.jsonl")))
done_ids = get_shard_ids(shards)

for shard_id in trange(NUM_SHARDS, desc="Generating Question-Answer pairs for all shards."):
    if shard_id in done_ids:
        continue
    shard_ds = ds.shard(num_shards=NUM_SHARDS, index=shard_id)
    passages = [row["content"] for row in shard_ds]
    passages = tqdm(passages, desc=f"Generating for shard {shard_id}")
    res = list(generate_dataset(passages, model, tokenizer))
    out_path = str(shards_folder.joinpath(f"shard_{shard_id}.jsonl"))
    save_jsonl(out_path, res)
    done_ids.add(shard_id)
    print("Pushing all shards to repo.")
    push_all_shards(shards_folder, repo="kardosdrur/dawiki_qa_zephyr")

print("DONE")
