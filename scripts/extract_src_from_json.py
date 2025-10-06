import argparse
import json

from skxoss.util import extract_src_from_json_trace

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    json_data = json.load(open(args.json_path, "r"))
    extract_src_from_json_trace(json_data, args.save_path)