# Cross-Lingual word embeddings in the context of Ukrainian-English translation

> Project from Linear Algebra, which aims to study how linear algebra methods can help connect Ukrainian and English word embeddings.

### Installation
1. Clone the repository
```
git clone https://github.com/okqsna/word-embeddings-la.git
```
2. Download Ukrainian and English embedding models from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)
- Create a folder `model` in the main directory and place the unarchived files with embedding models there.

Or
- Use tools from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) after downloading the requirements.
```
import fasttext
import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')
fasttext.util.download_model('uk', if_exists='ignore')

ft_eng = fasttext.load_model('cc.en.300.bin')
ft_uk = fasttext.load_model('cc.uk.300.bin')
```

### Usage
#### 1. Activate Python virtual environment
- For POSIX
```
python -m venv venv
source venv/bin/activate
```
- For Windows
```
python -m venv venv
venv\Scripts\activate
```
#### 2. Install the libraries from the requirements.
```
pip install -r requirements.txt
```
### Project structure

```
word-embeddings-la/
├── data/                 # All data from MUSE
│   ├── original/         # Original data
│   └── usage/            # Cleaned and processed data
├── model/                # FastText embedding models
├── scripts/              # Python scripts for experiments
│   ├── alignment.py
│   ├── data_preprocessing.py
│   └── main_process.ipynb
└── requirements.txt      # Python dependencies
```
### Videos with explanation
1. [Video from Oksana Moskviak](https://www.youtube.com/watch?v=wnyQZj9yCpY&t=1s)
2. [Video from Olena Dovbenchuk](https://www.youtube.com/watch?v=TkA_asXaBEE&feature=youtu.be)
3. [Video from Alina Bodnar](https://www.youtube.com/watch?v=0psD0UzPHMM&feature=youtu.be)

### Contributors
- [Oksana Moskviak](https://github.com/okqsna)
- [Olena Dovbenchuk](https://github.com/Olenadovb)
- [Alina Bodnar](https://github.com/alinabodnarpn)


<I>@ Created by UCU APPS students</i> 
