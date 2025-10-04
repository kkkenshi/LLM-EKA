import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')


def bio_tagging(text, entities):
    """
    Convert a sentence and its entities into BIO format.

    Args:
        text (str): The text of the sentence.
        entities (list): List of entities in "Type:Value" format.

    Returns:
        list of tuples: (word, BIO-tag)
        list: entities that were not matched in the sentence
    """
    words = word_tokenize(text)
    words_lower = [word.lower() for word in words]  # lowercase for matching
    tags = ['O'] * len(words)  # initialize all tags as 'O'
    unmatched_entities = []  # store entities that were not matched

    for entity in entities:
        entity = entity.strip()
        if ':' not in entity:
            # Skip entities without type specification
            unmatched_entities.append(entity)
            continue

        # Split entity into type and value, and clean up whitespace
        entity_type, entity_value = entity.split(':', 1)
        entity_type = entity_type.strip()
        entity_value = entity_value.strip()
        entity_value = entity_value.replace('@', '')
        entity_value = re.sub(r'#\S+', '', entity_value).strip()

        entity_words = word_tokenize(entity_value.lower())
        if not entity_words:
            unmatched_entities.append(entity_value)
            continue

        entity_length = len(entity_words)
        found = False  # flag to check if entity is found in the sentence

        # Search for the entity in the text
        for i in range(len(words_lower) - entity_length + 1):
            if words_lower[i:i + entity_length] == entity_words:
                found = True
                tags[i] = f"B-{entity_type}"
                for j in range(1, entity_length):
                    tags[i + j] = f"I-{entity_type}"
        if not found:
            unmatched_entities.append(entity_value)

    return list(zip(words, tags)), unmatched_entities


def process_entities_field(entities_field):
    """
    Preprocess the replaced_entities field to ensure each entity has a type.

    Args:
        entities_field (str): Raw replaced_entities string

    Returns:
        list: entities in "Type:Value" format
    """
    raw_entities = [e.strip() for e in entities_field.split(',')]
    entities = []
    last_type = None
    for e in raw_entities:
        if ':' in e:
            last_type, value = e.split(':', 1)
            last_type = last_type.strip()
            value = value.strip()
            entities.append(f"{last_type}:{value}")
        else:
            # If type is missing, use the previous entity type
            if last_type is not None:
                entities.append(f"{last_type}:{e}")
            else:
                entities.append(e)
    return entities


def convert_to_bio(input_file_path, output_file_path):
    """
    Convert a CSV file with generated content and replaced_entities
    into BIO format.

    Args:
        input_file_path (str): CSV file path with 'generated_content' and 'replaced_entities' columns
        output_file_path (str): Path to save the output BIO file

    Returns:
        int: Number of sentences converted
    """
    df = pd.read_csv(input_file_path)
    converted_sentences_count = 0

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Remove hashtags from generated content
            generated_content = re.sub(r'#\S+', '', row['generated_content'])
            # Preprocess entities
            entities = process_entities_field(row['replaced_entities'])

            bio_tags, unmatched = bio_tagging(generated_content, entities)

            if any(tag != 'O' for _, tag in bio_tags):
                f.write("\n".join(f"{word} {tag}" for word, tag in bio_tags))
                f.write("\n\n")
                converted_sentences_count += 1
            else:
                print(f"No entities in: {generated_content}")
                print(f"Unmatched entities: {unmatched}")

    return converted_sentences_count


if __name__ == "__main__":
    # Input and output file paths
    input_file_path = ""
    output_file_path = ""

    converted_count = convert_to_bio(input_file_path, output_file_path)
    print(f"Converted sentences: {converted_count}")
