import codecs
import json


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_path',
        default='data/test.jsonl',
        type=str,
        help='path to input jsonl file'
    )
    parser.add_argument(
        '-s', '--select_type',
        default='longest',
        choices=['longest', 'norm'],
        type=str,
        help='criteria to select names for entities'
    )
    args = parser.parse_args()

    docs: dict = load_json(args.input_path)
    output_path = args.input_path.replace(
        ".json", f".names.{args.select_type}.txt"
    )
    select(docs, output_path, args.select_type)


def load_json(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def select(docs: dict, output_path: str, select_type: str):
    with open(output_path, "w") as f_out:
        for doc in docs.values():
            men_id_to_mention: dict = doc["mentions"]
            for ent_id, entity in doc["entities"].items():
                if select_type == "longest":
                    mention = select_longest_mention(entity, men_id_to_mention)
                    entity_name = mention["text"]
                else:
                    entity_name = entity["normalized_name"]
                f_out.write(f'{entity_name}\n')


def select_longest_mention(entity, men_id_to_mention) -> dict:
    longest_mention_with_name = None
    longest_with_name = -1
    longest_mention = None
    longest = -1
    for mention_id in entity["member_mention_ids"]:
        mention = men_id_to_mention[mention_id]
        mention_text = mention["text"]
        if "NAME" in mention["entity_type"]:
            if longest_with_name < len(mention_text):
                longest_with_name = len(mention_text)
                longest_mention_with_name = mention
        if longest < len(mention_text):
            longest = len(mention_text)
            longest_mention = mention
    if longest_mention_with_name:
        return longest_mention_with_name
    return longest_mention


if __name__ == '__main__':
    main()
