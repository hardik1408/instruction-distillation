#!/bin/bash
set -e
#python zero_shot.py --model qwen
#python few_shot.py --model qwen --k 1
#python few_shot.py --model qwen --k 2
#python few_shot.py --model qwen --k 3
#python few_shot.py --model qwen --k 5
#python few_shot.py --model qwen --k 10
#python instruction_manual.py --model qwen

#python approachone.py --model qwen --k 1
#python approachone.py --model qwen --k 2
#python approachone.py --model qwen --k 3
#python approachone.py --model qwen --k 5
#python approachone.py --model qwen --k 10
#python approachone.py --model qwen --k 25

#python hybrid.py --model qwen --k 1
#python hybrid.py --model qwen --k 2
#python hybrid.py --model qwen --k 3
#python hybrid.py --model qwen --k 5
#python hybrid.py --model qwen --k 10

# python hybrid_new.py --model qwen --k 1
# python hybrid_new.py --model qwen --k 2
# python hybrid_new.py --model qwen --k 3
# python hybrid_new.py --model qwen --k 5
# python hybrid_new.py --model qwen --k 10


# python approachone.py --model gemma --k 50

# python generate_ablation_instructions.py --model gemma --instr_method caption
python ablation_b.py --model gemma --instr_method caption --k 1
python ablation_b.py --model gemma --instr_method caption --k 5
python ablation_b.py --model gemma --instr_method caption --k 10
python ablation_b.py --model gemma --instr_method caption --k 25



python generate_ablation_instructions.py --model gemma --instr_method paragraph
python ablation_b.py --model gemma --instr_method paragraph --k 1
python ablation_b.py --model gemma --instr_method paragraph --k 5
python ablation_b.py --model gemma --instr_method paragraph --k 10
python ablation_b.py --model gemma --instr_method paragraph --k 25
