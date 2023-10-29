# !/bin/sh

# Single target attr (gender) with vicreg
echo "--- run 1"
python pipeline.py --seed "1" --model "vicreg" --target-idx "20"
echo "--- run 2"
python pipeline.py --seed "2" --model "vicreg" --target-idx "20"
echo "--- run 3"
python pipeline.py --seed "3" --model "vicreg" --target-idx "20"
echo "--- run 4"
python pipeline.py --seed "4" --model "vicreg" --target-idx "20"
echo "--- run 5"
python pipeline.py --seed "5" --model "vicreg" --target-idx "20"

# Multiple target attr with vicreg
echo "--- run 6"
python pipeline.py --seed "1" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 7"
python pipeline.py --seed "2" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 8"
python pipeline.py --seed "3" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 9"
python pipeline.py --seed "4" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 10"
python pipeline.py --seed "5" --model "vicreg" --target-idx "2, 19, 39" --sample-iter 900


# Single target attr (gender) with dino
echo "--- run 11"
python pipeline.py --seed "1" --model "dino" --target-idx "20"
echo "--- run 12"
python pipeline.py --seed "2" --model "dino" --target-idx "20"
echo "--- run 13"
python pipeline.py --seed "3" --model "dino" --target-idx "20"
echo "--- run 14"
python pipeline.py --seed "4" --model "dino" --target-idx "20"
echo "--- run 15"
python pipeline.py --seed "5" --model "dino" --target-idx "20"

# Multiple target attr with dino
echo "--- run 16"
python pipeline.py --seed "1" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 17"
python pipeline.py --seed "2" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 18"
python pipeline.py --seed "3" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 19"
python pipeline.py --seed "4" --model "dino" --target-idx "2, 19, 39" --sample-iter 900
echo "--- run 20"
python pipeline.py --seed "5" --model "dino" --target-idx "2, 19, 39" --sample-iter 900