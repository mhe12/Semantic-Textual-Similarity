#!/bin/bash
echo "Assuming Paragram-Phrase XXL and  Paragram-SL999 has been downloaded were put in ./datasets+scoring_script folder"

sudo pip install pexpect unidecode
git clone git://github.com/dasmith/stanford-corenlp-python.git
cd stanford-corenlp-python
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip
unzip stanford-corenlp-full-2014-08-27.zip
cd ..

python -m nltk.downloader stopwords wordnet
sudo pip install jsonrpclib
git clone https://github.com/ma-sultan/monolingual-word-aligner.git
cd stanford-corenlp-python
rm -rf corenlp.py
cp ../corenlp.py ./
python corenlp.py