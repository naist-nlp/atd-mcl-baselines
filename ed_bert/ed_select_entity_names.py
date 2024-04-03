import jsonlines


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

    select(args.input_path, args.select_type)


def select(input_path: str, select_type: str):
    output_path = input_path.replace(
        ".jsonl", f".names.{select_type}.txt"
    )
    f_out = open(output_path, "w")
    with jsonlines.open(input_path, "r") as reader:
        for doc in reader:
            men_id_to_mention: dict = doc["mentions"]
            for ent_id, entity in doc["entities"].items():
                if select_type == "longest":
                    mention = select_longest_mention(entity, men_id_to_mention)
                    entity_name = mention["text"]
                else:
                    entity_name = entity["normalized_name"]
                f_out.write(f'{entity_name}\n')
    f_out.close()


def select_longest_mention(entity, men_id_to_mention) -> dict:
    longest_mention_with_name = None
    longest_with_name = -1
    longest_mention = None
    longest = -1
    for mention_id in entity["member_mention_ids"]:
        mention = men_id_to_mention[mention_id]
        mention_text = mention["text"]
        if "NAME" in mention["entity_label"]:
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
