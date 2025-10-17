import csv
import ast
import time
import openai
import random
import os
import json

API_SECRET_KEY = "" #your api_key
BASE_URL = ""
SLEEP_INTERVAL = 0.2
RESET_INTERVAL = 0.2

input_folder = r"/data"
output_folder = r"/data"
os.makedirs(output_folder, exist_ok=True)


def get_completion_with_retry(session, model="gpt-3.5-turbo", retries=3):
    openai.api_key = API_SECRET_KEY
    openai.api_base = BASE_URL
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=session,
                temperature=1,
            )
            return response['choices'][0]['message']['content'], response['usage']
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(RESET_INTERVAL)
    return None, None


def extract_entities(labels):
    entities = []
    try:
        labels = ast.literal_eval(labels)
        for label in labels:
            entity_type = label.get('type', 'Unknown')
            entity_value = label.get('value', '')
            entities.append(f"{entity_type}:{entity_value}")
    except (SyntaxError, ValueError):
        pass
    return entities

with open("entity_lists.json", "r", encoding="utf-8") as f:
    ENTITY_LISTS = json.load(f)

Vaccine_related = ENTITY_LISTS.get("Vaccine-related", [])
symptoms = ENTITY_LISTS.get("Symptom", [])
Drugs = ENTITY_LISTS.get("Drug", [])
Disease = ENTITY_LISTS.get("Disease", [])
Organization = ENTITY_LISTS.get("Organization", [])
Person = ENTITY_LISTS.get("Person", [])
Location = ENTITY_LISTS.get("Location", [])

csv_files = ['METS-CoV-train-origin.csv']
for csv_file in csv_files:
    input_file_path = os.path.join(input_folder, csv_file)
    output_file_path = os.path.join(output_folder, f"output_{csv_file}")

    data = []
    with open(input_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)

    total_tokens = 0

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['tweet_content', 'entities', 'generated_content', 'replaced_entities']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(data):
            sentence = item['tweet_content']
            entities = extract_entities(item['labels'])
            if not entities:
                print(f"No entities found in sentence {idx + 1}/{len(data)}. Skipping...")
                continue

            print(f"Processing sentence {idx + 1}/{len(data)} in file {csv_file}: {sentence}")
            print(f"Extracted entities: {entities}")

            entity_replacement_lists = {
                'Vaccine-related': Vaccine_related,
                'Symptom': symptoms,
                'Drug': Drugs,
                'Disease': Disease,
                'Organization': Organization,
                'Person': Person,
                'Location': Location,
            }

            replaced_entities_info = {}
            for entity in entities:
                entity_type, entity_value = entity.split(':', 1)
                replaced_entities_info.setdefault(entity_type, []).append(entity_value)

            for i in range(10):
                temp_replaced_entities_info = replaced_entities_info.copy()

                for entity_type, entity_values in temp_replaced_entities_info.items():
                    if entity_type in entity_replacement_lists:
                        available_entities = list(set(entity_replacement_lists[entity_type]) - set(entity_values))
                        if available_entities:
                            temp_replaced_entities_info[entity_type] = random.sample(available_entities,
                                                                                     len(entity_values))
                        else:
                            temp_replaced_entities_info[entity_type] = entity_values

                replaced_entities_descriptions = []
                for entity_type, names in temp_replaced_entities_info.items():
                    description = f"the {entity_type.lower()} entities {', '.join(names[:-1])} and {names[-1]}" if len(
                        names) > 1 else f"the {entity_type.lower()} entity {names[0]}"
                    replaced_entities_descriptions.append(description)

                replaced_entities_description = ", ".join(replaced_entities_descriptions)
                system_message = {
                    "role": "system",
                    "content": f'Take the sentence as an example "{sentence}", please generate a new covid-19 tweet which only has {replaced_entities_description}, without introducing any other named entity.'
                }

                session = [system_message]
                response, usage = get_completion_with_retry(session)
                if response and usage:
                    total_tokens += usage['total_tokens']
                    replaced_entities_str = ', '.join(
                        [f"{k}:{', '.join(v)}" for k, v in temp_replaced_entities_info.items()])
                    print(f"Generated response {i + 1}: {response}")
                    print(f"Replaced entities: {temp_replaced_entities_info}")

                    writer.writerow({
                        "tweet_content": sentence,
                        "entities": ', '.join(entities),
                        "generated_content": response,
                        "replaced_entities": replaced_entities_str,
                    })
                else:
                    print(f"Failed to get response {i + 1} from GPT. Skipping...")

                time.sleep(SLEEP_INTERVAL)

    print(f"Processed file: {csv_file}, output written to {output_file_path}")
    print(f"Total tokens used for {csv_file}: {total_tokens}")
