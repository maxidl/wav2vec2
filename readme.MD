# Work in Progress

wav2vec2 finetuning on common voice german.

## General Notes & Caveats
- I made some changes to the [run_common_voice.py script](https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/run_common_voice.py) improving support of large datasets (`de` language is resource intensive).
- preprocessing and finetuning are split into separate scripts.
- a big advantage is that after preprocessing once, the finetuning has a low startup time.
- includes improvements for `group_by_length` problem, see [forum post](https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5)
- all arguments are configured in `args.json`
- argument classes are moved to `argument_classes.py`.


## Preprocessing
- `prepare_dataset.py` handles all the preprocessing.
- It produces a directory `./resampled` containing float32 tensor representations the resampled and processed audio. For `de` data this requires ~96GB disk space.
- Runtime: ~4 hours on 32 threads (configure via `preprocessing_num_workers` argument)
- RAM requirement: <5GB

## Training
- ToDo


## Environment Setup with conda
This is what I ran to create my env:
```
conda create -n wav2vec python=3.8
conda install -c pytorch -c nvidia pytorch torchaudio cudatoolkit=11.1
pip install transformers datasets
pip install jiwer==2.2.0
pip install lang-trans==0.6.0
pip install librosa==0.8.0
```
