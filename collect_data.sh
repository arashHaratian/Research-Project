# !/bin/sh

# Single target attr (gender) with vicreg
echo "--- run 1"
python3 pipeline.py --seed "1" --model "vicreg" --target-idx "20"
echo "--- run 2"
python3 pipeline.py --seed "2" --model "vicreg" --target-idx "20"
echo "--- run 3"
python3 pipeline.py --seed "3" --model "vicreg" --target-idx "20"
echo "--- run 4"
python3 pipeline.py --seed "4" --model "vicreg" --target-idx "20"
echo "--- run 5"
python3 pipeline.py --seed "5" --model "vicreg" --target-idx "20"

# Multiple target attr with vicreg
echo "--- run 6"
python3 pipeline.py --seed "1" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 7"
python3 pipeline.py --seed "2" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 8"
python3 pipeline.py --seed "3" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 9"
python3 pipeline.py --seed "4" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 10"
python3 pipeline.py --seed "5" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900


# Single target attr (gender) with dino
echo "--- run 11"
python3 pipeline.py --seed "1" --model "dino" --target-idx "20"
echo "--- run 12"
python3 pipeline.py --seed "2" --model "dino" --target-idx "20"
echo "--- run 13"
python3 pipeline.py --seed "3" --model "dino" --target-idx "20"
echo "--- run 14"
python3 pipeline.py --seed "4" --model "dino" --target-idx "20"
echo "--- run 15"
python3 pipeline.py --seed "5" --model "dino" --target-idx "20"

# Multiple target attr with dino
echo "--- run 16"
python3 pipeline.py --seed "1" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 17"
python3 pipeline.py --seed "2" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 18"
python3 pipeline.py --seed "3" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 19"
python3 pipeline.py --seed "4" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 20"
python3 pipeline.py --seed "5" --model "dino" --target-idx "2, 19, 39" --sample-iter 900