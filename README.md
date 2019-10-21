# IN4334 - Analytics and Machine Learning for Software Engineering
Analytics and Machine Learning for Software Engineering

## Preprocessing
First install the required dependencies:

```
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk
pip install --upgrade orjson
```

Then start a Python shell and run the following commands:
```
import nltk
nltk.download('punkt')
```

Now, make the necessary modifications to the configuration in `preprocessing/constants.py`. Then, from the root of this repo, run `python -m preprocessing.main`. The script will first index the dataset, which can take a couple of minutes, and then process the dataset, which can take more than an hour.
