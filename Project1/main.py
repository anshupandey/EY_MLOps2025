import pandas as pd
import mlflow, mlflow.sklearn
import argparse

def main():
    mlflow.sklearn.autolog()

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data",dest='training_data', type=str)
    parser.add_argument("--max_depth",dest='max_depth', type=int)

    args = parser.parse_args()
    df = pd.read_csv(args.training_data)
    max_depth = args.max_depth
    print("data loaded for  shape : ",df.shape)
    print("argement max_depth is ",max_depth)
    return df.shape

if __name__=="__main__":
    main()
