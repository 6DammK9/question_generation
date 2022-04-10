# Project Code for 2021-22 Semester B CS 6493 Project Group 10

This repository is forked from [patil-suraj/question_generation](https://github.com/patil-suraj/question_generation). Multiple issues has been fixed, along with additoinal feature, to  provide comparasion between supervised / unsupervised SOTA models in question generation task.

## Environment requirement

- Instead of `requirement.txt`, we use [miniconda](https://docs.conda.io/en/latest/miniconda.html) for setup.

```cmd
# python=3.10 will lead to almost no avaliable packages!
conda create -n sklearn-env -c conda-forge scikit-learn python=3.9

conda install -c conda-forge jupyterlab_widgets
conda install -c conda-forge ipywidgets

conda install -c pytorch pytorch
conda install -c pytorch torchvision
conda install -c pytorch torchaudio

conda install -c pytorch torchtext
conda install -c conda-forge ray-tune

# 2.7.0 will have serious issue (cudnn issue)
# 2.6.0 has some weird bug (e.g. keras not found)
conda install -c conda-forge tensorflow-gpu=2.5.0

conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-image
conda install -c conda-forge scipy
conda install -c conda-forge networkx

conda install -c conda-forge datasets
conda install -c conda-forge spacy==3.0.0

# 3.0.2 has incapability with Python 3.9. Hence transformers must be newer then 4.0
conda install -c conda-forge transformers
conda install -c conda-forge nltk
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

## Findings / Added features

- As in what you see in 🤗 model hub, this pipeline will eventually feed the formatted input to the model. However it was originally done in a unsupervised manner (using `t5-small-qa-qg-hl` to extract answers *per sentence*), therefore the code is a bit more difficult to read.
- Extract answers example: [t5-small-qa-qg-hl](https://huggingface.co/valhalla/t5-small-qa-qg-hl) (Original answer was [`Serbian`, `1856`, `1943`, `alternating current`]) (as in notebook `zz1c`)

```txt
> extract answers: <hl> Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system. <hl> </s>
> икола есла<sep>
```

- Workaround for the case above (answer is not found in context). I further break the word to "hope for" some generated contents. In default case, empty string will be returned. It crashes in original pipeline.
- Then "answer extraction" can be bypassed by providing **sentence with highlight** with corrosponding highlight token (`<hl>` or `[HL]`) (`</s>` is optional)
- Then the pipeline now supports both BART and T5 (from 2 repos).
- However the actual implementation of BERT is different from online version [bart-squad-qg-hl](https://huggingface.co/p208p2002/bart-squad-qg-hl) (Online result = `What nationality was Nikola Tesla?`):

```txt
> Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a [HL]Serbian[HL] American inventor, electrical engineer, mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system.
> What was the nationality of Konnikola tesla? 
```

## TODOs

- To support more repos, token handling should be based from model name instead of model type. However currently popular / avaliable SOTA models for this tasks are BART and T5 only, [Re-current BERT](https://aclanthology.org/D19-5821/) was "[stuck in implementation](https://github.com/p208p2002/Transformer-QG-on-SQuAD/issues/1)", and [ERNIE-GEN](https://arxiv.org/abs/2001.11314) uses a different framework which has no pyTorch / 🤗 adaptation yet.
- `zz2`. The expected trainning dataset is already incompatable. Follow guides for [this repo](https://github.com/p208p2002/Transformer-QG-on-SQuAD) instead.
- SQuAD v2.0 support. Plausable answers can be cast directly for QG task, but it is not effective when the trainning task is stuck.

## Notebook list

- `zz1`: Original notebook in repo. **Clear**.
- `zz1b`: Some corner case which may crash the original model. **Clear**.
- `zz1c`: Fusing with BART, which is not completed in original repo. Now both BART and T5 can be in supervised mode. **Clear**.
- `zz2`: Training and retrive score metric. **In progress**.
- `zz3`: Minimal `e2e-qg` with score metric. **Clear**.
- `zz3b`: Minimal `question-generation` with SQuAD dataset. **Clear**.
- `zz3c` (4x): Full SQuAD validation set on small / base model. **Clear**.
- `zz4a`: BART base with supervised highlighted answer. **Clear**.
- `zz4b`: T5 base with supervised highlighted answer. **Clear**.

## Results

As claimed by both repos, all models are trained with SQuAD v1.1. "Base model" is included.
Different from forked version. Score is based on full validation set of SQuAD v1.1 in 🤗datasets (formaerly `nlp`) *per context*:

- `hyp.txt`: Concatenated generated questions.
- `ref1.txt`: Original questions.
- `ref2.txt`: Original concext.
Note that the score is generally higher then what you've seen in web. Their performance should be identical.

| Name                                                                    | Highlight     | BLEU-1  | BLEU-2  | BLEU-4  | METEOR  | ROUGE-L |
|-------------------------------------------------------------------------|---------------|---------|---------|---------|---------|---------|
| [t5-base-e2e-qg](https://huggingface.co/valhalla/t5-base-e2e-qg)        | Supervised    | 68.6667 | 53.0235 | 33.7465 | 28.5125 | 32.7107 |
| [bart-squad-qg-hl](https://huggingface.co/p208p2002/bart-squad-qg-hl)   | Supervised    | 67.0877 | 51.0051 | 31.2478 | 26.7013 | 31.6968 |
| [t5-base-e2e-qg](https://huggingface.co/valhalla/t5-base-e2e-qg)        | Unsupervised  | 57.8001 | 47.8133 | 34.1749 | 19.0514 | 35.0973 |
| [t5-base-qg-hl](https://huggingface.co/valhalla/t5-base-qg-hl)          | Unsupervised  | 69.8286 | 53.4806 | 34.1254 | 21.7064 | 34.8645 |
| [t5-small-e2e-qg](https://huggingface.co/valhalla/t5-small-e2e-qg)      | Unsupervised  | 53.2628 | 43.6088 | 30.6282 | 17.7136 | 33.5326 |
| [t5-small-qg-hl](https://huggingface.co/valhalla/t5-small-qg-hl)        | Unsupervised  | 69.4194 | 53.1734 | 33.8424 | 21.2269 | 34.0925 |

## Citations

```txt
@misc{questiongeneration20,
    author = {Philip Huang},
    title = {Question Generation},
    publisher = {GitHub},
    journal = {GitHub repository},
    year = {2021},
    howpublished={\url{https://github.com/p208p2002/Transformer-QG-on-SQuAD}}
}
@misc{questiongeneration20,
    author = {Suraj Patil},
    title = {Question Generation},
    publisher = {GitHub},
    journal = {GitHub repository},
    year = {2020},
    howpublished={\url{https://github.com/patil-suraj/question_generation}}
}
```

## Relevant papers

- https://arxiv.org/abs/1906.05416
- https://www.aclweb.org/anthology/D19-5821/
- https://arxiv.org/abs/2005.01107v1
