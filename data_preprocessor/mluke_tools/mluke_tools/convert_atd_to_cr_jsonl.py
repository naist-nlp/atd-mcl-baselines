import json
import os
from typing import Any, Dict, Iterator, List, Tuple, Union


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))

    with open(args.output, mode="w", encoding="utf-8") as f:
        for doc_id, sentences in read_atd(args.input):
            document = {"id": doc_id, "examples": sentences}
            f.write(encoder.encode(document))
            f.write("\n")


def read_atd(
    file: Union[str, bytes, os.PathLike]
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    with open(file, mode="r", encoding="utf-8") as f:
        for doc_id, doc in json.load(f).items():
            mentions = {
                k: {
                    "start": v["span"][0],
                    "end": v["span"][1],
                    "label": v["entity_type"],
                    "entity_id": v["entity_id"],
                }
                for k, v in doc["mentions"].items()
            }

            examples: List[Dict[str, Any]] = []
            for sentence_id, sentence in sorted(doc["sentences"].items()):
                example = {
                    "id": f"{doc_id}:{sentence_id}",
                    "text": sentence["text"],
                    "mentions": [mentions[mid] for mid in sentence["mention_ids"]],
                }
                example["mentions"] = [
                    m for m in example["mentions"] if m["entity_id"] is not None
                ]
                examples.append(example)

            yield doc_id, examples


if __name__ == "__main__":
    main()
