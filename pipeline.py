import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="vicreg", type=str)
    parser.add_argument('-b', '--batch-size', default="vicreg", type=int)
    parser.add_argument('-s', '--sample-size', default="vicreg", type=int)
    parser.add_argument('-f', '--file', default="./pipeline_results.txt", type=str)
    parser.add_argument('-t', '--target-idx', default="20", type=str)


    return parser.parse_args()


def load_model():
    pass

def read_data():
    pass

def save_data():
    pass






if __name__ == '__main__':

    args = argparser()
    
    