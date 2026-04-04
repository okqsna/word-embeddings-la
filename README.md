# Cross-Lingual word embeddings in the context of Ukrainian-English translation

> Project from Linear Algebra, which aims to study how linear algebra methods can help connect Ukrainian and English word embeddings.

### Installation
1. Clone the repository
```
git clone https://github.com/okqsna/word-embeddings-la.git
```
2. Download Ukrainian and Engish embedding models from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)

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
│   ├── data_preprocessing.py
│   └── main_process.ipynb
└── requirements.txt      # Python dependencies
```

### Contributors
- [Oksana Moskviak](https://github.com/okqsna)
- [Olena Dovbenchuk](https://github.com/Olenadovb)
- [Alina Bodnar](https://github.com/alinabodnarpn)


<I>@ Created by UCU APPS students</i> 
