# IN4334 - Analytics and Machine Learning for Software Engineering
Analytics and Machine Learning for Software Engineering

## Preprocessing
First install the required dependencies:

```bash
pip install -r requirements.txt
```

Or with pipenv:

```bash
pipenv install
pipenv shell
```

And download the spacy model:
```bash
python -m spacy download en_core_web_sm
```

Then start a Python shell and run the following commands:
```python
import nltk
nltk.download('punkt')
```

Now, make the necessary modifications to the configuration in `preprocessing/constants.py`. Then, from the root of this repo, run `python -m preprocessing.main`. The script will first index the dataset, which can take a couple of minutes, and then process the dataset, which can take more than an hour.
