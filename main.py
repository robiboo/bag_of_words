import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.pipeline import Pipeline


def main():
  data = pd.read_csv('./data.csv')
  data = data.fillna(0)

  train = data.sample(frac=0.7)
  test = data.drop(train.index)

if __name__ == "__main__":
  main()
