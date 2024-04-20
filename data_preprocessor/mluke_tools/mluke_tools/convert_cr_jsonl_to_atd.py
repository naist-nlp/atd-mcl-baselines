import json
import os
from typing import Any, Dict, Iterator, Tuple, Union


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--pretty", "-p", action="store_true")
    args = parser.parse_args()

    documents = {doc_id: doc for doc_id, doc in read_jsonl(args.input)}
    with open(args.output, mode="w", encoding="utf-8") as f:
        options = {"indent": 2} if args.pretty else {}
        json.dump(documents, f, ensure_ascii=False, **options)


def read_jsonl(
    file: Union[str, bytes, os.PathLike]
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            sentences: Dict[str, Any] = {}
            mentions: Dict[str, Any] = {}
            entities: Dict[str, Any] = {}

            for example in doc["examples"]:
                sentence_id = example["id"].split(":")[-1]
                text = example["text"]
                mention_ids = []

                for mention in example["mentions"]:
                    mention_id = f"M{len(mentions) + 1:03}"
                    mentions[mention_id] = {
                        "sentence_id": sentence_id,
                        "entity_id": mention["entity_id"],
                        "span": [mention["start"], mention["end"]],
                        "text": text[mention["start"] : mention["end"]],
                        "entity_type": mention["label"],
                    }
                    mention_ids.append(mention_id)

                    entity_id = mention["entity_id"]
                    assert entity_id is not None
                    if entity_id not in entities:
                        entities[entity_id] = {"member_mention_ids": []}
                    entities[entity_id]["member_mention_ids"].append(mention_id)

                sentences[sentence_id] = {"text": text, "mention_ids": mention_ids}

            output = {
                "sentences": sentences,
                "mentions": mentions,
                "entities": entities,
            }
            yield doc["id"], output


if __name__ == "__main__":
    main()
