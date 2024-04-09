import jsonlines


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_path',
        default='data/osm/20230620_all_extnames.txt',
        type=str,
        help='path to file that contains extnames'
    )
    parser.add_argument(
        '-o', '--output_path',
        default=None,
        type=str,
        help='path to output jsonl file'
    )
    args = parser.parse_args()

    if args.output_path is None:
        output_path = args.input_path.replace(".txt", ".jsonl")
    else:
        output_path = args.output_path
    print(f'Save {output_path}')

    convert_txt_to_jsonl(args.input_path, output_path)


def convert_txt_to_jsonl(input_path: str, output_path: str) -> None:
    writer = jsonlines.open(output_path, "w")
    num_entries = 0

    with open(input_path, "r") as f:
        for line in f:
            line = line.rstrip().split("\t")
            assert len(line) == 3

            entry = {"id": line[1]}
            for elem in line[1].split("|"):
                key, value = elem.split("=")
                entry[key] = value.rstrip()

            if entry["name"]:
                entry["members"] = line[2].split(",")
                writer.write(entry)
                num_entries += 1
            else:
                print(f"Name is empty: {str(entry)}")

    writer.close()
    print(f"{num_entries} entries have been saved in {output_path}")


if __name__ == '__main__':
    main()
