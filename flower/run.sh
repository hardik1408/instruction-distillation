#!/bin/bash
set -e

# python zero_shot.py --model qwen
# python few_shot.py --model qwen --k 1
# python few_shot.py --model qwen --k 2
python few_shot.py --model qwen --k 3
python few_shot.py --model qwen --k 5
python few_shot.py --model qwen --k 10
python instruction_manual.py --model qwen

python approachone.py --model qwen --k 1
python approachone.py --model qwen --k 2
python approachone.py --model qwen --k 3
python approachone.py --model qwen --k 5
python approachone.py --model qwen --k 10
python approachone.py --model qwen --k 25

python hybrid.py --model qwen --k 1
python hybrid.py --model qwen --k 2
python hybrid.py --model qwen --k 3
python hybrid.py --model qwen --k 5
python hybrid.py --model qwen --k 10

python hybrid_new.py --model qwen --k 1
python hybrid_new.py --model qwen --k 2
python hybrid_new.py --model qwen --k 3
python hybrid_new.py --model qwen --k 5
python hybrid_new.py --model qwen --k 10


# python approachone.py --model gemma --k 50

# python eval.py --p hybrid/gemma/predictions_hybrid_k5_kp1.json --o hybrid/gemma/metrics_5_1.json
# python eval.py --p hybrid/gemma/predictions_hybrid_k5_kp2.json --o hybrid/gemma/metrics_5_2.json
# python eval.py --p hybrid/gemma/predictions_hybrid_k5_kp3.json --o hybrid/gemma/metrics_5_3.json
# python eval.py --p hybrid/gemma/predictions_hybrid_k10_kp1.json --o hybrid/gemma/metrics_10_1.json
# python eval.py --p hybrid/gemma/predictions_hybrid_k10_kp2.json --o hybrid/gemma/metrics_10_2.json
# python eval.py --p hybrid/gemma/predictions_hybrid_k10_kp3.json --o hybrid/gemma/metrics_10_3.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k1_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k1_kp2.json --o hybrid_new/gemma/metrics_1_2.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
# python eval.py --p hybrid_new/gemma/predictions_hybrid_k5_kp1.json --o hybrid_new/gemma/metrics_1_1.json
