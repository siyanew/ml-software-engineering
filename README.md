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
=======
### Train
* Training from config: `python train.py --config config/<config.json>`
* Training from checkpoint: `python train.py --resume saved/models/<model_name>/<start_time>`
* Analyse the logs with Tensorboard: `tensorboard --logdir saved/log/<model_name>`

Note that when resuming a model from a checkpoint, the corresponding `config.json` from `saved/models/<model_name>/<start_time>` will be used. 
  

### Test
* Test from config: `python test.py --config config/<config.json>`
* Test from checkpoint: `python test.py --resume saved/models/<model_name>/<start_time>`
* Analyse the test logs with Tensorboard: `tensorboard --logdir saved/test_log/<model_name>`

The test script will compute the following on the test set: 
* The loss and perplexity.
* Inference on the diff data. The file with predictions of the commit messages are stored with the `.pred` suffix. 
For the exact files location see `config['inference']`.

