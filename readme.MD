# Work in Progress

## conda env
```
conda create -n wav2vec python=3.8
conda install -c pytorch -c nvidia pytorch torchaudio cudatoolkit=11.1
pip install transformers datasets
pip install jiwer==2.2.0
pip install lang-trans==0.6.0
pip install librosa==0.8.0

```



## Caveats

- preprocessing and finetuning are split into separate scripts.
- `prepare-dataset.py` takes ~5 hours using 32 threads. It saves raw tensors (float32, ~96GB) of the fully preprocessed audio to disk.
- new argument: `datasets_in_memory` (bool). whether to load full datasets into memory before training. If you set this to false, also set `group_by_length` to false (see [this issue](https://github.com/huggingface/transformers/issues/10909).




