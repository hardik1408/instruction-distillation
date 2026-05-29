#!/bin/bash
set -e

# python zero_shot.py --model qwen
# python few_shot.py --model qwen --k 1
# python few_shot.py --model qwen --k 2
# python few_shot.py --model qwen --k 3
# python few_shot.py --model qwen --k 5
# python few_shot.py --model qwen --k 10
# python instruction_manual.py --model qwen

# python hybrid.py --model qwen --k 1
# python hybrid.py --model qwen --k 5
# python hybrid.py --model qwen --k 10

# python approachone.py --model qwen --k 1
# python approachone.py --model qwen --k 2
# python approachone.py --model qwen --k 3
# python approachone.py --model qwen --k 5
# python approachone.py --model qwen --k 10
# python approachone.py --model qwen --k 25
# python approachone.py --model gemma --k 50

python hybrid_new.py --model qwen --k 1
python hybrid_new.py --model qwen --k 2
python hybrid_new.py --model qwen --k 3
python hybrid_new.py --model qwen --k 5
python hybrid_new.py --model qwen --k 10


