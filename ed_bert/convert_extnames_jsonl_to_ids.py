import jsonlines


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_file',
        default='data/20230620_all_extnames.jsonl',
        type=str,
        help='path to file that contains extnames'
    )
    args = parser.parse_args()

    extract_names(args.input_file)


def extract_names(input_path: str) -> None:
    output_path = input_path.replace(".jsonl", ".ids.txt")
    f_out = open(output_path, "w")
    with jsonlines.open(input_path, "r") as reader:
        for entry in reader:
            f_out.write(f'{entry["id"]}\n')
    f_out.close()
    print(f"Entries are saved in {output_path}")


if __name__ == '__main__':
    main()
