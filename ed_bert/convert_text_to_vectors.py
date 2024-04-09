from bert_tokenizers import ATDTokenizer
from transformers import BertModel
from vectorizers import Vectorizer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_path',
        default='data/test.names.longest.txt',
        type=str,
        help='path to file that contains span names')
    parser.add_argument(
        '-b', '--batch_size',
        default=128,
        type=int,
        help='batch size')
    parser.add_argument(
        '-l', '--layer',
        default=12,
        type=int,
        help='extracting hidden states at n-th layer')
    parser.add_argument(
        '-s', '--data_size',
        default=None,
        type=int,
        help='data size')
    parser.add_argument(
        '-bert', '--bert_path',
        default='model/cl-tohoku/bert-base-japanese-whole-word-masking',
        type=str,
        help='bert model to be used')
    parser.add_argument(
        '-c', '--context',
        action='store_true'
    )
    parser.add_argument(
        '-subword', '--only_subword',
        action='store_true'
    )
    args = parser.parse_args()
    bert_path = args.bert_path

    tokenizer = ATDTokenizer.from_pretrained(bert_path, flush=True)
    print("Tokenizer loaded from {}".format(bert_path))
    print("do_word_tokenize: %s" % str(tokenizer.do_word_tokenize))
    print("word_tokenizer_type: %s" % tokenizer.word_tokenizer_type)
    print("lower_case: %s" % str(tokenizer.lower_case))
    print("do_subword_tokenize: %s" % str(tokenizer.do_subword_tokenize))
    print("subword_tokenizer_type: %s" % tokenizer.subword_tokenizer_type)

    model = BertModel.from_pretrained(
        bert_path, output_hidden_states=True
    )
    bert_name = bert_path.split("/")[-1]
    print("BERT Model loaded from {}".format(bert_path), flush=True)

    subword_output_path = args.input_path.replace(
        ".txt", f".subwords.{bert_name}.jsonl"
    )
    vector_output_path = subword_output_path.replace(".jsonl", ".vecs.hdf5")
    knn_output_path = vector_output_path.replace(".hdf5", ".knn.txt")

    vectorizer = Vectorizer(tokenizer, model)

    if args.context:
        spans, span_indices, contexts = vectorizer.load_spans_with_context(
            args.input_file
        )
        if args.data_size:
            span_indices = span_indices[:args.data_size]
            print(f"- {len(span_indices)} spans will be used")
        subword_tokens, subword_indices = vectorizer.convert_spans_with_context_to_subwords(
            span_indices, contexts, subword_output_path
        )
    else:
        spans: list[str] = vectorizer.load_spans(args.input_path)
        if args.data_size:
            spans = spans[:args.data_size]
            print(f"- {len(spans)} spans will be used")
        subword_tokens = vectorizer.convert_spans_to_subwords(
            spans, subword_output_path
        )
        subword_indices = [[0, len(tokens)] for tokens in subword_tokens]

    if args.only_subword:
        print("Completed subwords. Exits.")
        exit()

    vectorizer.convert_subwords_to_vectors(
        subword_tokens, subword_indices,
        batch_size=args.batch_size, n_th_layer=args.layer,
        output_path=vector_output_path
    )

    vectorizer.check_knn(
        spans=spans[:1000], vec_path=vector_output_path,
        output_path=knn_output_path
    )


if __name__ == '__main__':
    main()
