# nlp-cheatsheets

## Usage

#### split_corpus.sh
Split given corpus into random train/test subsets. `TRAIN_RATIO` represent the proportion of the dataset to include in the train split.

```shell
$ scripts/split_corpus.sh [INPUT_FILE] [OUTPUT_FILE] [TRAIN_RATIO]

# example
$ scripts/split_corpus.sh sample/zero_to_hundred.txt sample/zero_to_hundred 0.8
Split corpus into train(#80) and test(#20) corpus, respectively.
```