# Work in Progress

wav2vec2 finetuning on common voice german.

## General Notes & Caveats
I made some changes to the [run_common_voice.py script](https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/run_common_voice.py) improving support of large datasets (`de` language is resource intensive):

- preprocessing and finetuning are split into separate scripts, `prepare_dataset.py` and `run_finetuning.py`.
- The focus here is on running the preprocessing once, and having a short training startup time afterwards.
- argument classes are moved to `argument_classes.py`.


## Preprocessing
- `prepare_dataset.py` handles all the preprocessing.
- It saves the dataset containing only the paths and labels to disk, as an arrow table in parquet format
- It loads, resamples, processes and saves audio files into raw tensors, instead of HF dataset.
- It produces a directory `./resampled` containing float32 tensor representations the resampled and processed audio.
- It saves the processor to disk, for reuse in training script (it is not deterministic across scripts)
- Runtime: ~4 hours on 32 threads (configure via `preprocessing_num_workers` argument) for `de`
- RAM requirement: <5GB for `de`
- Storage Requirement: For `de` data the `./resampled` dir requires ~96GB disk space.

## Training
- `run_finetuning.py` does the training. Use `'finetune.sh`, `finetune_distributed.sh` or `args.json` to specify all the arguments.
- Custom dataset to load the output of `preprocess_dataset.py`.
- includes improvements for `group_by_length` problem, see [forum post](https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5). The dataset is iterated in parallel by multiple workers to quickly get the lengths of each input sequence.
- includes a less aggressively smoothed trainer progress bar for a better epoch time estimate.

- RAM requirement: <15GB on full `de`
- One epoch `de` takes ~14 hours on a single RTX 3090.
- I did not see good enough speedup from distributed training (tried two gpus). 


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
