# Wav2Vec2-Large-XLSR-53-German

Fine-tuned facebook/wav2vec2-large-xlsr-53 on German using the Common Voice dataset.
[Model in Hugginface Model Hub](https://huggingface.co/maxidl/wav2vec2-large-xlsr-german).

## General Notes & Caveats
I made some changes to the [run_common_voice.py script](https://github.com/huggingface/transformers/blob/master/examples/research_projects/wav2vec2/run_common_voice.py) improving support of large datasets (`de` language is resource intensive):

- preprocessing and finetuning are split into separate scripts, `prepare_dataset.py` and `run_finetuning.py`.
- The focus here is on running the preprocessing once, and having a short training startup time afterwards.
- argument classes are moved to `argument_classes.py`.


## Preprocessing
- `prepare_dataset.py` handles all the preprocessing.
- It saves the dataset containing only the paths and labels to disk, as an arrow table in parquet format
- It loads, resamples, processes and saves audio files into raw tensors, instead of HF dataset.
- It produces a directory `./resampled` containing float32 tensor representations of the resampled and processed audio.
- It saves the processor to disk, for reuse in training script (it is not deterministic across scripts)
- Runtime: ~4 hours on 32 threads (configure via `preprocessing_num_workers` argument) for `de`
- RAM requirement: <5GB for `de`
- Storage Requirement: For `de` data the `./resampled` dir requires ~96GB disk space.

## Training
- `run_finetuning.py` does the training. Use `'finetune.sh`, `finetune_distributed.sh` or `args.json` to specify all the arguments.
- Custom dataset to load the output of `preprocess_dataset.py`.
- includes improvements for `group_by_length` problem, see [forum post](https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5). The dataset is iterated in parallel by multiple workers to quickly get the lengths of each input sequence.
- Limits the maximum sequence length during trainin to the 99% quantile to save gpu memory and allow for larger batch sizes.
- includes a less aggressively smoothed trainer progress bar for a better training time estimate.

- RAM requirement: while training itself takes <15GB on full `de`, the evaluation during training still requires ~170GB RAM. ToDo: compute WER in chunks.
- The model was trained for 50k steps, taking around 30 hours on a single A100.
- I did not see good enough speedup from distributed training (tried two gpus), most likely due to gradient checkpointing not being supported in Distributed Data Parallel training.


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


## Reproducing the Model
- run preprocessing as in `preprocess.sh`
- run training as in `finetune.sh`


# Model Card

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on German using the [Common Voice](https://huggingface.co/datasets/common_voice) dataset.
When using this model, make sure that your speech input is sampled at 16kHz.

## Usage
When using this model, make sure that your speech input is sampled at 16kHz.

The model can be used directly (without a language model) as follows:

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "de", split="test[:8]") # use a batch of 8 for demo purposes

processor = Wav2Vec2Processor.from_pretrained("maxidl/wav2vec2-large-xlsr-german")
model = Wav2Vec2ForCTC.from_pretrained("maxidl/wav2vec2-large-xlsr-german") 

resampler = torchaudio.transforms.Resample(48_000, 16_000)

"""
Preprocessing the dataset by:
- loading audio files
- resampling to 16kHz
- converting to array
- prepare input tensor using the processor
"""
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

# run forward
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"])
"""
Example Result:

Prediction: [
    'zieh durch bittet draußen die schuhe aus',
    'es kommt zugvorgebauten fo',
    'ihre vorterstrecken erschienen it modemagazinen wie der voge karpes basar mariclair',
    'fürliepert eine auch für manachen ungewöhnlich lange drittelliste',
    'er wurde zu ehren des reichskanzlers otto von bismarck errichtet',
    'was solls ich bin bereit',
    'das internet besteht aus vielen computern die miteinander verbunden sind',
    'der uranus ist der siebinteplanet in unserem sonnensystem s'
]

Reference: [
    'Zieht euch bitte draußen die Schuhe aus.',
    'Es kommt zum Showdown in Gstaad.',
    'Ihre Fotostrecken erschienen in Modemagazinen wie der Vogue, Harper’s Bazaar und Marie Claire.',
    'Felipe hat eine auch für Monarchen ungewöhnlich lange Titelliste.',
    'Er wurde zu Ehren des Reichskanzlers Otto von Bismarck errichtet.',
    'Was solls, ich bin bereit.',
    'Das Internet besteht aus vielen Computern, die miteinander verbunden sind.',
    'Der Uranus ist der siebente Planet in unserem Sonnensystem.'
]
"""
```


## Evaluation

The model can be evaluated as follows on the German test data of Common Voice:


```python
import re
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

"""
Evaluation on the full test set:
- takes ~20mins (RTX 3090).
- requires ~170GB RAM to compute the WER. A potential solution to this is computing it in chunks.
  See https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5 on how to implement this.
"""
test_dataset = load_dataset("common_voice", "de", split="test") # use "test[:1%]" for 1% sample
wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("maxidl/wav2vec2-large-xlsr-german")
model = Wav2Vec2ForCTC.from_pretrained("maxidl/wav2vec2-large-xlsr-german")
model.to("cuda")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
	batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
	inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

	with torch.no_grad():
		logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

	pred_ids = torch.argmax(logits, dim=-1)
	batch["pred_strings"] = processor.batch_decode(pred_ids)
	return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8) # batch_size=8 -> requires ~14.5GB GPU memory

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
# WER: 12.615308
```

**Test Result**: 12.62 %


## Training

The Common Voice German `train` and `validation` were used for training.
The script used for training can be found [here](https://github.com/maxidl/wav2vec2).
The model was trained for 50k steps, taking around 30 hours on a single A100.

The arguments used for training this model are:
```
python run_finetuning.py \
--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
--dataset_config_name="de" \
--output_dir=./wav2vec2-large-xlsr-german \
--preprocessing_num_workers="16" \
--overwrite_output_dir \
--num_train_epochs="20" \
--per_device_train_batch_size="64" \
--per_device_eval_batch_size="32" \
--learning_rate="1e-4" \
--warmup_steps="500" \
--evaluation_strategy="steps" \
--save_steps="5000" \
--eval_steps="5000" \
--logging_steps="1000" \
--save_total_limit="3" \
--freeze_feature_extractor \
--activation_dropout="0.055" \
--attention_dropout="0.094" \
--feat_proj_dropout="0.04" \
--layerdrop="0.04" \
--mask_time_prob="0.08" \
--gradient_checkpointing="1" \
--fp16 \
--do_train \
--do_eval \
--dataloader_num_workers="16" \
--group_by_length
```
