import argparse
from typing import Tuple


NOT_ENTITY = 'O'


def get_spans_from_BIO_seq(
        words: list[str],
        labels: list[str],
) -> list[Tuple[int, int, str]]:

    spans = []
    span_begin = -1
    char_idx = 0
    prev_tag  = None
    prev_cate = None
    for word, label in zip(words, labels):
        if label != NOT_ENTITY:
            tag, cate = label.split('-')

            if prev_cate == None: # None VS B/I-X
                span_begin = char_idx
                prev_tag   = tag
                prev_cate  = cate

            elif prev_cate != cate or tag == 'B': # B/I-X VS B/I-Y or B-X
                spans.append((span_begin, char_idx, prev_cate))
                span_begin = char_idx
                prev_tag   = tag
                prev_cate  = cate

            elif tag == 'I':    # B/I-X VS B-X
                pass

        else:
            if prev_cate != None: # B/I-X VS None
                spans.append((span_begin, char_idx, prev_cate))
                span_begin = -1
                prev_tag   = None
                prev_cate  = None

        char_idx += len(word)

    if prev_cate is not None:
        len_sen = len(''.join([w for w in words]))
        spans.append((span_begin, len_sen, prev_cate))

    return spans


def read_and_write(
        input_path: str,
        output_path: str,
        doc_id: str = None,
) -> None:

    sen_id = 0
    sen    = None
    words  = []
    labels = []

    with (open(input_path, encoding='utf-8') as f,
          open(output_path, 'w', encoding='utf-8') as fw,
    ):
        for line in f:
            line = line.rstrip('\n')

            if line.startswith('#'):
                sen_id += 1
                continue

            elif not line:
                spans = get_spans_from_BIO_seq(words, labels)
                sen = ''.join(words)
                spans_str = ';'.join([f'_,{span[0]}:{span[1]},{span[2]}' for span in spans])
                fw.write(f'{doc_id}\t{sen_id}\t{sen}\t{spans_str}\n')

                sen = None
                words = []
                labels = []
                
            else:
                array = line.split('\t')
                word  = array[1]
                attrs = array[9].split('|')
                ene = NOT_ENTITY
                for attr in attrs:
                    if attr.startswith('ENE'):
                        ene = attr[4:]
                words.append(word)
                labels.append(ene)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', '-i',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--doc_id', '-d',
        type=str,
    )
    args = parser.parse_args()

    read_and_write(args.input_path, args.output_path, doc_id=args.doc_id)
