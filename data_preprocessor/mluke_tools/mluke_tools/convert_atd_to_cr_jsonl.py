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
            examples: List[Dict[str, Any]] = []
            for sentence_id, sentence in sorted(doc["sentences"].items()):
                mentions = []
                for mid in sentence["mention_ids"]:
                    mention = doc["mentions"][mid]
                    if (
                        mention["entity_id"] is None
                        or mention.get("generic", False)
                        or mention.get("ref_spec_amb", False)
                    ):
                        continue
                    mentions.append(
                        {
                            "start": mention["span"][0],
                            "end": mention["span"][1],
                            "label": mention["entity_type"],
                            "entity_id": mention["entity_id"],
                        }
                    )
                example = {
                    "id": f"{doc_id}:{sentence_id}",
                    "text": sentence["text"],
                    "mentions": mentions,
                }
                examples.append(example)

            yield doc_id, examples


if __name__ == "__main__":
    main()
