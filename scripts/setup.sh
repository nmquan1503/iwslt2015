pip install setuptools
pip install sacrebleu
pip install rouge-score
pip uninstall -y \
    unbabel-comet \
    transformers \
    tokenizers \
    sentencepiece \
    entmax
pip install unbabel-comet

git clone https://github.com/nmquan1503/attention.git -b main -q
cd attention
pip install . --no-build-isolation -q
cd ..
