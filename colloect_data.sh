# !/bin/sh



python pipeline.py --seed "1" --model "vicreg" --target-idx "20"
python pipeline.py --seed "2" --model "vicreg" --target-idx "20"
python pipeline.py --seed "3" --model "vicreg" --target-idx "20"
python pipeline.py --seed "4" --model "vicreg" --target-idx "20"
python pipeline.py --seed "5" --model "vicreg" --target-idx "20"

python pipeline.py --seed "1" --model "vicreg" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "2" --model "vicreg" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "3" --model "vicreg" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "4" --model "vicreg" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "5" --model "vicreg" --target-idx "3, 7, 13, 20"

python pipeline.py --seed "1" --model "dino" --target-idx "20"
python pipeline.py --seed "2" --model "dino" --target-idx "20"
python pipeline.py --seed "3" --model "dino" --target-idx "20"
python pipeline.py --seed "4" --model "dino" --target-idx "20"
python pipeline.py --seed "5" --model "dino" --target-idx "20"

python pipeline.py --seed "1" --model "dino" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "2" --model "dino" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "3" --model "dino" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "4" --model "dino" --target-idx "3, 7, 13, 20"
python pipeline.py --seed "5" --model "dino" --target-idx "3, 7, 13, 20"