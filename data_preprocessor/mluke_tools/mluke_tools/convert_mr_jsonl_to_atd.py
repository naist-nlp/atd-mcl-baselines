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

            for example in doc["examples"]:
                sentence_id = example["id"].split(":")[-1]
                text = example["text"]
                mention_ids = []

                for entity in example["entities"]:
                    mention_id = f"M{len(mentions) + 1:03}"
                    mentions[mention_id] = {
                        "sentence_id": sentence_id,
                        "entity_id": None,
                        "span": [entity["start"], entity["end"]],
                        "text": text[entity["start"] : entity["end"]],
                        "entity_type": entity["label"],
                    }
                    mention_ids.append(mention_id)

                sentences[sentence_id] = {"text": text, "mention_ids": mention_ids}

            yield doc["id"], {"sentences": sentences, "mentions": mentions}


if __name__ == "__main__":
    main()
