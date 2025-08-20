# !/bin/bash

python main.py --pre_model_name=LPA --activity_type=A6 --dataset=data_sample && echo "LPA A6 Launched." &
P0=$!

python main.py --pre_model_name=LPA --activity_type=A3 --dataset=data_sample && echo "LPA A3 Launched." &
P1=$!

python main.py --pre_model_name=LPA --activity_type=None --dataset=data_sample && echo "LPA None Launched." &
P2=$!


wait $P0 $P1 $P2