import jsonlines
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertModel, BertJapaneseTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to be used: {}".format(DEVICE), flush=True)

if DEVICE.type == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Vectorizer(object):

    def __init__(
            self, tokenizer: BertJapaneseTokenizer, model: BertModel
    ):
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def load_spans(span_path) -> list[str]:
        print(f"Load spans from {span_path}", flush=True)
        with open(span_path, "r") as f:
            spans = [line.rstrip("\n") for line in f]
        print(f"- {len(spans)} spans have been loaded")
        return spans

    @staticmethod
    def load_spans_with_context(span_path):
        print(f"Load spans with context from {span_path}", flush=True)
        spans = []
        span_indices = []
        contexts = []
        with open(span_path, "r") as f:
            for line in f:
                line = line.rstrip("\n").split("\t")
                spans.append(line[0])
                begin, end = line[1].split(" ")
                span_indices.append([int(begin), int(end)])
                contexts.append(line[2])
        print(f"- {len(spans)} spans have been loaded")
        return spans, span_indices, contexts

    def convert_spans_to_subwords(
            self, spans: list[str], output_path: str
    ) -> list[list[str]]:
        print("Convert spans to subwords", flush=True)
        subwords = []
        with jsonlines.open(output_path, "w") as writer:
            for span_text in spans:
                subword_tokens: list[str] = self.tokenizer.tokenize(span_text)
                writer.write(
                    {"span_text": span_text,
                     "subword_tokens": subword_tokens}
                )
                subwords.append(subword_tokens)
        print(f"- {len(subwords)} subwords have been saved in {output_path}")
        return subwords

    def convert_spans_with_context_to_subwords(
            self, span_indices: list[list[int, int]],
            contexts: list[str], output_path: str
    ) -> tuple[list[list[str]], list[list[int, int]]]:
        print("Convert spans to subwords", flush=True)
        all_subwords = []
        all_indices = []
        with jsonlines.open(output_path, "w") as writer:
            for [begin, end], context in zip(span_indices, contexts):
                span_text = context[begin: end]
                subwords, indices = self.tokenizer.create_tokens_and_span(
                    span_txt=span_text, span_begin=begin, span_end=end,
                    context=context
                )

                writer.write(
                    {"span_text": span_text,
                     "span_subword_indices": indices,
                     "context_subword_tokens": subwords}
                )
                all_subwords.append(subwords)
                all_indices.append(indices)
        print(f"- {len(all_subwords)} subwords have been saved in {output_path}")
        return all_subwords, all_indices

    def convert_subwords_to_vectors(
            self,
            subword_tokens: list[list[str]],
            subword_indices: list[list[int, int]],
            batch_size: int, n_th_layer: int, output_path: str
    ) -> None:
        print("Convert subwords to vectors", flush=True)
        f_out = h5py.File(output_path, "w")
        span_index = 0
        batch_index = 0
        tokens_list = []
        indices_list = []
        prev_time = time.time()

        for tokens, indices in zip(subword_tokens, subword_indices):
            tokens_list.append(tokens)
            indices_list.append(indices)

            if len(tokens_list) == batch_size:
                batch_index += 1
                batch = self.create_each_batch(tokens_list)
                hidden_states_list = self.convert_batch_to_hidden_states(
                    batch, n_th_layer
                )

                for hidden_states, (begin, end) in zip(hidden_states_list,
                                                       indices_list):
                    # +1 for begin & end because of [CLS] added to tokens
                    vec = l2_normalize(
                        np.mean(hidden_states[begin + 1: end + 1], axis=0)
                    )
                    f_out.create_dataset(
                        name=f'{span_index}', dtype='float32', data=vec
                    )
                    span_index += 1
                tokens_list = []
                indices_list = []

                if batch_index % 100 == 0:
                    cur_time = time.time()
                    print("{} batches|{:.2f}sec".format(
                        batch_index, cur_time - prev_time
                    ), flush=True)
                    prev_time = cur_time

        if tokens_list:
            batch = self.create_each_batch(tokens_list)
            hidden_states_list = self.convert_batch_to_hidden_states(
                batch, n_th_layer
            )

            for hidden_states, (begin, end) in zip(hidden_states_list,
                                                   indices_list):
                vec = l2_normalize(
                    np.mean(hidden_states[begin + 1: end + 1], axis=0)
                )
                f_out.create_dataset(
                    name=f'{span_index}', dtype='float32', data=vec
                )
                span_index += 1

        f_out.close()
        assert len(subword_tokens) == span_index
        print(f"- {span_index} vectors have been saved in {output_path}")

    def create_each_batch(self, tokens_list: list[list[str]]) -> dict:
        max_length = max([len(tokens) for tokens in tokens_list]) + 2
        max_length = min(max_length, 512)
        batch = defaultdict(list)
        for tokens in tokens_list:
            encoding = self.tokenizer.create_ids(
                tokens, max_length=max_length
            )
            for k, v in encoding.items():
                batch[k].append(v)
        return {k: torch.tensor(v).to(DEVICE) for k, v in batch.items()}

    def convert_batch_to_hidden_states(
            self, batch: dict, n_th_layer: int
    ) -> list:
        with torch.no_grad():
            outputs = self.model(**batch)
        # (batch size, num tokens, 768)
        return outputs[2][n_th_layer].cpu().numpy().tolist()

    @staticmethod
    def check_knn(spans: list[str], vec_path: str, output_path) -> None:
        print("Trial KNN search", flush=True)

        f_hdf5 = h5py.File(vec_path, "r")
        query_vecs = [
            f_hdf5[str(span_index)]
            for span_index, span in enumerate(spans)
        ]
        db_vecs = np.asarray(query_vecs)
        assert len(spans) == len(query_vecs)

        f_knn = open(output_path, "w")
        for span, query_vec in zip(spans, query_vecs):
            f_knn.write(f'- Query: {span}\n')
            dots = np.dot(db_vecs, np.asarray(query_vec))
            arg_sort = np.argsort(dots)[::-1]
            for rank, arg in enumerate(arg_sort[:10]):
                f_knn.write(
                    f'-- {rank + 1} {spans[arg]} {dots[arg]}\n'
                )
            f_knn.write("\n")
        f_hdf5.close()

        print(f"- KNN results have been saved in {output_path}")


def extract_bert_hidden_states(
        input_ids: Tensor,
        attention_mask: Tensor,
        model: BertModel,
        n_th_layer: int
) -> np.ndarray:
    with torch.no_grad():
        # outputs: (last hidden states, pooler outputs, hidden states)
        # last hidden states: (batch size, num tokens, 768)
        # hidden states: (num layers, batch size, num tokens, 768)
        outputs = model(input_ids, attention_mask)
    # 0 = embedding layer
    # 1-12 = bert layers
    hidden_states = outputs[2][n_th_layer][0]
    vecs = hidden_states.numpy()
    return vecs


def l2_normalize(x):
    z = np.linalg.norm(x, ord=2)
    return x / z if z != 0 else x * 0


def compute_phrase_vector(
        vecs: np.ndarray,
        begin: int, end: int,
        normalize=False
) -> np.ndarray:
    subword_vecs = vecs[begin: end]
    vec = np.mean(subword_vecs, axis=0)
    if normalize:
        return l2_normalize(vec)
    return vec


def create_phrase_vector(
        span_txt: str,
        span_begin: int,
        span_end: int,
        context: str,
        tokenizer: BertJapaneseTokenizer,
        model: BertModel
) -> np.ndarray:
    tokens, subword_span = tokenizer.create_tokens_and_span(
        span_txt, span_begin, span_end, context
    )
    # [CLS]と[SEP]も自動的に追加される
    encoding = tokenizer.create_ids(tokens, max_length=512)
    encoding = {k: torch.tensor([v]) for k, v in encoding.items()}
    hidden_states: np.ndarray = extract_bert_hidden_states(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        model=model, n_th_layer=12
    )
    return compute_phrase_vector(
        hidden_states,
        begin=subword_span[0]+1,   # +1 は先頭の[CLS]の追加の分
        end=subword_span[1]+1,
        normalize=True
    )


def extract_batch_hidden_states(
        spans: list[dict],
        tokenizer: BertJapaneseTokenizer,
        model: BertModel,
        output_path: str,
        batch_size: int,
        n_th_layer: int,
        test_knn=True
) -> None:
    print("Convert data loader to BERT hidden states", flush=True)
    f_out = h5py.File(output_path, "w")
    entry_index = -1
    batch_index = 0
    tokens_list = []
    prev_time = time.time()

    for span in spans:
        tokens_list.append(span["subword_tokens"])

        if len(tokens_list) == batch_size:
            batch_index += 1
            batch = create_each_batch(tokens_list, tokenizer)
            hidden_states_list = convert_batch_to_hidden_states(
                batch, model, n_th_layer
            )

            for tokens, hidden_states in zip(tokens_list, hidden_states_list):
                vec = l2_normalize(np.mean(hidden_states[1: -1], axis=0))
                entry_index += 1
                f_out.create_dataset(
                    name=f'{entry_index}', dtype='float32', data=vec
                )
            tokens_list = []

            if batch_index % 100 == 0:
                cur_time = time.time()
                print("{} batches|{:.2f}sec".format(
                    batch_index, cur_time - prev_time
                ), flush=True)
                prev_time = cur_time

    if tokens_list:
        batch = create_each_batch(tokens_list, tokenizer)
        hidden_states_list = convert_batch_to_hidden_states(
            batch, model, n_th_layer
        )

        for tokens, hidden_states in zip(tokens_list, hidden_states_list):
            vec = l2_normalize(np.mean(hidden_states[1: -1], axis=0))
            entry_index += 1
            f_out.create_dataset(
                name=f'{entry_index}', dtype='float32', data=vec
            )

    f_out.close()
    print(f"Save {entry_index+1} vectors in {output_path}")

    if test_knn:
        print("Test KNN search", flush=True)

        os.makedirs("outputs", exist_ok=True)
        f_knn = open("outputs/test_knn.txt", "w")
        print("- Results saved in outputs/test_knn.txt")

        f_hdf5 = h5py.File(output_path, "r")
        vecs = [
            f_hdf5[str(entry_index)]
            for entry_index, span in enumerate(spans[:100])
        ]
        db_entry_vecs = np.asarray(vecs)

        for span, query_vec in zip(spans, vecs):
            f_knn.write(f'Query: {span["span_txt"]}\n')
            dots = np.dot(db_entry_vecs, np.asarray(query_vec))
            arg_sort = np.argsort(dots)[::-1]
            for rank, arg in enumerate(arg_sort[:10]):
                f_knn.write(
                    f'- {rank + 1} {spans[arg]["span_txt"]} {dots[arg]}\n'
                )
            f_knn.write("\n")

        f_hdf5.close()


def convert_spans_to_tokens(
        spans: list[dict],
        tokenizer: BertJapaneseTokenizer
) -> tuple[list, list]:
    print("Convert spans to subword tokens", flush=True)
    all_tokens = []
    all_subword_spans = []
    for span in spans:
        tokens, subword_span = tokenizer.create_tokens_and_span(
            span_txt=span["txt"],
            span_begin=span["begin"],
            span_end=span["end"],
            context=span["context"]
        )
        all_tokens.append(tokens)
        all_subword_spans.append(subword_span)
    return all_tokens, all_subword_spans


def convert_tokens_to_data_loader(
        all_tokens: list[list[str]],
        tokenizer: BertJapaneseTokenizer,
        batch_size: int
) -> DataLoader:
    print("Convert tokens to data loader", flush=True)
    # 「+2」は[CLS]と[SEP]の分
    max_length = max([len(tokens) for tokens in all_tokens]) + 2
    max_length = min(max_length, 512)
    print(f"- Max tokens: {max_length}")

    data_for_loader: list[dict] = []
    for tokens in all_tokens:
        encoding = tokenizer.create_ids(tokens, max_length=max_length)
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        data_for_loader.append(encoding)
    return DataLoader(
        data_for_loader, batch_size=batch_size, shuffle=False
    )


def create_each_batch(
        tokens_list: list, tokenizer: BertJapaneseTokenizer
) -> dict:
    max_length = max([len(tokens) for tokens in tokens_list]) + 2
    max_length = min(max_length, 512)
    batch = defaultdict(list)
    for tokens in tokens_list:
        encoding = tokenizer.create_ids(tokens, max_length=max_length)
        for k, v in encoding.items():
            batch[k].append(v)
    return {k: torch.tensor(v).to(DEVICE) for k, v in batch.items()}


def convert_batch_to_hidden_states(
        batch: dict, model: BertModel, n_th_layer: int
) -> list:
    with torch.no_grad():
        outputs = model(**batch)
    # (batch size, num tokens, 768)
    return outputs[2][n_th_layer].cpu().numpy().tolist()


def convert_tokens_to_hidden_states(
        all_tokens: list[list[str]],
        tokenizer: BertJapaneseTokenizer,
        model: BertModel,
        n_th_layer: int,
        batch_size: int
) -> list[np.ndarray]:
    print("Convert data loader to BERT hidden states", flush=True)
    all_hidden_states = []
    prev_time = time.time()
    batch_index = 0
    tokens_list = []

    for tokens in all_tokens:
        tokens_list.append(tokens)

        if len(tokens_list) == batch_size:
            batch_index += 1
            batch = create_each_batch(tokens_list, tokenizer)
            tokens_list = []
            all_hidden_states += convert_batch_to_hidden_states(
                batch, model, n_th_layer
            )

            if batch_index % 100 == 0:
                cur_time = time.time()
                print("{} batches|{:.2f}sec".format(
                    batch_index, cur_time - prev_time
                ), flush=True)
                prev_time = cur_time

    if tokens_list:
        batch = create_each_batch(tokens_list, tokenizer)
        all_hidden_states += convert_batch_to_hidden_states(
            batch, model, n_th_layer
        )

    return all_hidden_states


def convert_data_loader_to_hidden_states(
        dataloader: DataLoader,
        model: BertModel,
        n_th_layer: int,
) -> list[np.ndarray]:
    print("Convert data loader to BERT hidden states", flush=True)
    all_hidden_states = []
    prev_time = time.time()
    batch_index = 1

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            # outputs: (last hidden states, pooler outputs, hidden states)
            # last hidden states: (batch size, num tokens, 768)
            # hidden states: (num layers, batch size, num tokens, 768)
            outputs = model(**batch)
        # (batch size, num tokens, 768)
        hidden_states = outputs[2][n_th_layer]
        all_hidden_states += hidden_states.cpu().numpy().tolist()

        if batch_index % 100 == 0:
            cur_time = time.time()
            print("{} batches|{:.2f}sec".format(
                batch_index, cur_time - prev_time
            ), flush=True)
            prev_time = cur_time
        batch_index += 1
    # (num spans, num tokens, 768)
    return all_hidden_states
