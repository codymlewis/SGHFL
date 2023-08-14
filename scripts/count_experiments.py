#!/usr/bin/env python3

import sys
import json


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError()

    with open(f"configs/{sys.argv[1]}.json", 'r') as f:
        print(len(json.load(f)['experiments']))