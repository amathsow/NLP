Command for training the WMT14 English-French model
===================================================

Setup notes
-----------
- Data is obtained using the scripts in fairseq-py/data:
   https://github.com/facebookresearch/fairseq-py/tree/master/data 
- No further pre-processing
- Data is splitted in train and valid:
  - train contains 35760411 sentences
  - valid contains 26853 sentences
- We use a BPE sub-word vocublary with 40k tokens, trained jointly on the
  training data for both languages (see bpecodes)

```
DATADIR="/path/to/preprocessed/data"
python train.py \
  ${DATADIR} -a fconv_wmt_en_fr \
  --lr 1.0 --clip-norm 0.1 --dropout 0.1 --max-tokens 4000 \
  --force-anneal 30 --label-smoothing 0.1 
```
