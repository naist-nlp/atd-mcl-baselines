import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', required=True)
    parser.add_argument('--output_path', '-o', required=True)
    args = parser.parse_args()

    # for single json file contains multiple docs
    with open(args.input_path, encoding='utf-8') as f:
        data_old = json.load(f)

    data_new = {}
    for docid, ents in data_old.items():
        data_new[docid] = {'entities': ents}

    with open(args.output_path, 'w', encoding='utf-8') as fw:
        json.dump(data_new, fw, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
