# IN4334 - Analytics and Machine Learning for Software Engineering
This repository contains the source code for the course IN4334 - Analytics and Machine Learning for Software Engineering given at the Delft University of Technology.
With the help of deep learning, commit messages were generated based on `git diff` files with a Sequence to Sequence model.
For more information see the [paper](ML4SE_Final_Report_Group4.pdf).


## Data collection
Edit the config on the top of the git-helper/github_api.py file to change the language to gather or the amount of 
repositories.
```bash
pip install GitPython==3.0.3 PyGithub==1.43.8
python git-helper/github_api.py <output_dir>
```
After the scraping is finished you can find a 'msg' and 'diff' dir in your output_dir which will contain all the
messages and diffs collected from GitHub. You can supply the output_dir to the preprocessing script as argument.


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


## Training a model
* Training from config: `python train.py --config config/<config.json>`
* Training from checkpoint: `python train.py --resume saved/models/<subdirectories>/checkpoint.pth`
* Analyse the logs with Tensorboard: `tensorboard --logdir saved/log/<model_name>`

Note that when resuming a model from a checkpoint, the corresponding `config.json` from `saved/models/<model_name>/<subdirectories>` will be used. 

For GPU support for PyTorch with CUDA, see the official documentation [on the PyTorch site](https://pytorch.org/get-started/locally/)
  

## Testing a model
* Test from config: `python test.py --config config/<config.json>`
* Test from checkpoint: `python test.py --resume saved/models/<model_name>/<subdirectories>`
* Analyse the test logs with Tensorboard: `tensorboard --logdir saved/test_log/<model_name>`

The test script will compute the following on the test set: 
* The loss and perplexity.
* Inference on the diff data. The file with predictions of the commit messages are stored with the `.pred` suffix. 
For the exact files location see `config['inference']`.


## Existing implementation
The `OpenNMT-py` toolkit from [here](https://github.com/OpenNMT/OpenNMT-py) is included in this repository. It can be execute with the test and train scripts in `scripts` from the root of this repository with `./scripts/train.sh` or `./scripts/test.sh`


## Configs
The configuration used in this research are the following:
* Java data: `config/java.json`
* C# data: `config/cs.json`
* NMT1 from Jiang et al. : `config/nmt1.json`
* NMT1 but preprocessed with our preprocessing: `config/nmt1_preprocessed.json`


## Research artifacts (datasets, models, results)
The collected datasets, the trained model, and all of the testing results are available online at [Zenodo](https://zenodo.org/record/3519363)


## Resources
The following resources were used during creating of this codebase:
* https://github.com/victoresque/pytorch-template
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://bastings.github.io/annotated_encoder_decoder/
* https://github.com/bentrevett/pytorch-seq2seq
