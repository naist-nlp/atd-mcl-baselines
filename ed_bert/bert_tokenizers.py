from transformers import BertJapaneseTokenizer


class ATDTokenizer(BertJapaneseTokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tokens_and_span(
            self, span_txt: str, span_begin: int, span_end: int, context: str
    ):
        tokens = []
        prev_context = context[0: span_begin]
        if prev_context:
            tokens += self.tokenize(prev_context)
        subword_begin = len(tokens)

        if context:
            tokens += self.tokenize(context[span_begin: span_end])
        else:
            tokens += self.tokenize(span_txt)
        subword_end = len(tokens)

        post_context = context[span_end:]
        if post_context:
            tokens += self.tokenize(post_context)

        return tokens, [subword_begin, subword_end]

    def create_ids(self, tokens, max_length=None):
        input_ids = self.convert_tokens_to_ids(tokens)
        return self.prepare_for_model(
            input_ids,
            max_length=max_length,
            padding='max_length' if max_length else False,
            truncation=True if max_length else False
        )
