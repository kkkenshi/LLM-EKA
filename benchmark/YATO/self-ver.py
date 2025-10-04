
import openai
import time
from typing import List, Dict
from generate_entities import Vaccine_related, symptoms, Drugs, Disease, Organization, Person, Location
API_SECRET_KEY = ""
BASE_URL = ""

def verify_entities_batch(entities_batch: List[str], entity_type: str) -> Dict[str, str]:
    """
    Verify a batch of entities for a given entity type using GPT.

    Args:
        entities_batch: List of entity.
        entity_type: The domain of entities.

    Returns:
        Dictionary mapping each entity to 'Yes' or 'No'.
    """
    openai.api_key = API_SECRET_KEY
    openai.api_base = BASE_URL

    prompt = (
        f"Please determine whether each of the following entities is related to COVID-19 {entity_type}. "
        "Respond on each line in the format 'Entity: Yes' or 'Entity: No'.\n\n"
    )
    prompt += "\n".join(entities_batch)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message['content'].strip()
        validation_results = {}
        for line in answer.splitlines():
            if ':' in line:
                entity, result = line.split(":", 1)
                validation_results[entity.strip()] = result.strip()
        return validation_results

    except Exception as e:
        print(f"Error processing batch {entities_batch}: {e}")
        return {}


domains = {
    "Vaccine-related": Vaccine_related,
    "Symptom": symptoms,
    "Drug": Drugs,
    "Disease": Disease,
}

batch_size = 10
verified_entities_by_type = {}

for domain, entities in domains.items():
    verified_entities_by_type[domain] = []

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        results = verify_entities_batch(batch, domain)  # 动态传入领域

        for entity in batch:
            if results.get(entity, "No") == "Yes":
                verified_entities_by_type[domain].append(entity)

        time.sleep(1)

# 打印每个领域的已验证实体
for domain, verified in verified_entities_by_type.items():
    print(f"{domain} ({len(verified)}): {verified}")