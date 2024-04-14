import json
import os
from typing import Any, Dict, Iterator, List, Tuple, Union

import sudachipy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    tokenizer = sudachipy.Dictionary(dict="full").create(sudachipy.SplitMode.B)
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))

    with open(args.output, mode="w", encoding="utf-8") as f:
        for doc_id, sentences in read_atd(args.input, tokenizer):
            document = {"id": doc_id, "examples": sentences}
            f.write(encoder.encode(document))
            f.write("\n")


def read_atd(
    file: Union[str, bytes, os.PathLike], tokenizer=None
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    with open(file, mode="r", encoding="utf-8") as f:
        for doc_id, doc in json.load(f).items():
            mentions = {
                k: {
                    "start": v["span"][0],
                    "end": v["span"][1],
                    "label": v["entity_type"],
                }
                for k, v in doc["mentions"].items()
            }

            examples: List[Dict[str, Any]] = []
            for sentence_id, sentence in sorted(doc["sentences"].items()):
                text = sentence["text"]

                word_positions = None
                if tokenizer:
                    word_positions = [
                        (m.begin(), m.end()) for m in tokenizer.tokenize(text)
                    ]

                example = {
                    "id": f"{doc_id}:{sentence_id}",
                    "text": text,
                    "entities": [mentions[mid] for mid in sentence["mention_ids"]],
                    "word_positions": word_positions,
                }
                examples.append(example)

            yield doc_id, examples


if __name__ == "__main__":
    main()
