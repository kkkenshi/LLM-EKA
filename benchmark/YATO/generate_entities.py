import pandas as pd
import ast
import openai
import time
from typing import Dict, List, Set
import random
import math
import re
from typing import List
API_SECRET_KEY = ""#your api_key
BASE_URL = ""

def clean_entity_list(entities: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for e in entities:
        e_clean = re.sub(r'^\d+\.\s*', '', e)
        if e_clean and e_clean not in seen:
            cleaned.append(e_clean)
            seen.add(e_clean)
    return cleaned



def extract_entities_by_type(input_file_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(input_file_path, sep="\t")
    entity_types: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        labels = ast.literal_eval(row["labels"])
        for label in labels:
            entity_type = label.get("type", "Other")
            entity_value = label.get("value", "")
            if entity_type not in entity_types:
                entity_types[entity_type] = set()
            entity_types[entity_type].add(entity_value)
    return {etype: sorted(values) for etype, values in entity_types.items()}

def get_completion_with_retry(session, model="gpt-3.5-turbo", retries=3):
    openai.api_key = API_SECRET_KEY
    openai.api_base = BASE_URL
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=session,
                temperature=0.7,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None

def generate_new_entities(entity_type: str, examples: List[str], num: int = 10) -> List[str]:
    max_batch_size = 50
    total_batches = math.ceil(num / max_batch_size)
    all_new_entities = []

    for batch_idx in range(total_batches):
        batch_num = min(max_batch_size, num - batch_idx * max_batch_size)
        batch_examples = random.sample(examples, min(50, len(examples)))

        print(f"\n[Batch {batch_idx + 1}/{total_batches}] Generating {batch_num} entities for '{entity_type}'")
        print(f"Using examples: {', '.join(batch_examples)}")

        prompt = (
            f"There are some entities about COVID-19 {entity_type}, "
            f"such as {', '.join(batch_examples)}. "
            f"Please generate {batch_num} new entities of the same type. "
            f"Return them as a plain Python list of strings."
        )

        session = [{"role": "user", "content": prompt}]
        content = get_completion_with_retry(session)
        if not content:
            print(f"Batch {batch_idx + 1} failed: no response from GPT.")
            continue

        try:
            new_entities = ast.literal_eval(content)
        except Exception:
            new_entities = [line.strip("-• ") for line in content.splitlines() if line.strip()]

        print(f"Batch {batch_idx + 1} generated {len(new_entities)} entities.")
        all_new_entities.extend(new_entities)

    print(f"\nTotal new entities generated for '{entity_type}': {len(all_new_entities)}")
    return all_new_entities


def build_entity_lists():
    file_path = "METS-CoV-train-origin.csv"
    entities_by_type = extract_entities_by_type(file_path)
    entity_lists = {}
    for entity_type, entities in entities_by_type.items():
        new_entities = generate_new_entities(entity_type, entities, num=60)
        entity_lists[entity_type] = new_entities
    return entity_lists

# global
ENTITY_LISTS = build_entity_lists()
Vaccine_related = clean_entity_list(ENTITY_LISTS.get("Vaccine-related", []))
symptoms = clean_entity_list(ENTITY_LISTS.get("Symptom", []))
Drugs = clean_entity_list(ENTITY_LISTS.get("Drug", []))
Disease = clean_entity_list(ENTITY_LISTS.get("Disease", []))
Organization = clean_entity_list(ENTITY_LISTS.get("Organization", []))
Person = clean_entity_list(ENTITY_LISTS.get("Person", []))
Location = clean_entity_list(ENTITY_LISTS.get("Location", []))

