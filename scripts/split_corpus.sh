#shuffile and split corpus
INPUT_FILE=$1
OUTPUT_FILE=$2
TRAIN_RATIO=$3

ALL_LINE_NUM=`wc -l ${INPUT_FILE} | awk '{print $1}'`
TRAIN_BASE=`echo "scale=0; ${ALL_LINE_NUM}*${TRAIN_RATIO}/1" | bc` # TRAIN_BASE=`echo $ALL_LINE_NUM*$TRAIN_RATIO | bc`
TEST_BASE=`echo ${ALL_LINE_NUM}-${TRAIN_BASE} | bc`

shuf ${INPUT_FILE} > ${OUTPUT_FILE}
head -n ${TRAIN_BASE} ${OUTPUT_FILE} > "${OUTPUT_FILE}.train"
tail -n ${TEST_BASE} ${OUTPUT_FILE} > "${OUTPUT_FILE}.test"

echo "Split corpus into train(#${TRAIN_BASE}) and test(#${TEST_BASE}) corpus, respectively."
