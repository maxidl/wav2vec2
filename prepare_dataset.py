import json
import re
from pathlib import Path
import sys

from tqdm.auto import tqdm
import torch
import torchaudio
import pandas as pd
import pyarrow.parquet as pq

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from argument_classes import ModelArguments, DataTrainingArguments

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# increasing number of threads for torchaudio resample
print(f'Using {data_args.preprocessing_num_workers} threads')
torch.set_num_threads(data_args.preprocessing_num_workers)

# Get the datasets:
train_dataset = datasets.load_dataset(
    "common_voice", data_args.dataset_config_name, split=data_args.train_split_name
)
eval_dataset = datasets.load_dataset("common_voice", data_args.dataset_config_name, split="test")

# Create and save tokenizer
chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower() + " "
    return batch

train_dataset = train_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True, num_proc=data_args.preprocessing_num_workers)
eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True, num_proc=data_args.preprocessing_num_workers)

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
vocab_test = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

if data_args.max_train_samples is not None:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))

if data_args.max_val_samples is not None:
    eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

# Load pretrained tokenizer & create processor
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# =======================================================
# The following part is modified to:
#   - load, resample, process and save audio files into raw tensors
#   - process labels as before
#   - save datasets containing the paths and labels to disk, as arrow table in parquet format
#   - save processor to disk, for reuse in training script (I got a hint that it is not deterministic)

# load and resample audio, save as raw tensors
resampler = torchaudio.transforms.Resample(48_000, 16_000)
resampled_data_dir = Path('./resampled')
resampled_data_dir.mkdir(exist_ok=True)

def load_resample_save(f):
    f = Path(f)
    new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
    if not new_path.exists():
        speech_array, sampling_rate = torchaudio.load(f)
        speech_array_resampled = resampler(speech_array)
        input_values = processor(speech_array_resampled, sampling_rate=16_000).input_values
        input_values = torch.from_numpy(input_values).float().flatten()
        torch.save(input_values, new_path)
    return str(new_path)

print('load resample save')
new_train_paths = [load_resample_save(f) for f in tqdm(train_dataset['path'], miniters=100, desc='train')]
new_eval_paths = [load_resample_save(f) for f in tqdm(eval_dataset['path'], miniters=100, desc='eval')]

# update paths and sampling rate
train_dataset = train_dataset.map(
    lambda x: {'path': new_train_paths, 'sampling_rate':[16_000] * len(train_dataset), 'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    lambda x: {'path': new_eval_paths, 'sampling_rate':[16_000] * len(eval_dataset), 'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)

# tokenize targets
def tokenize_targets(batch):
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

print('preparing dataset: train')
train_dataset = train_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in train_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)
print('preparing dataset: eval')
eval_dataset = eval_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in eval_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)

# # save for disk, ready for training
pq.write_table(train_dataset.data, f'./{data_args.dataset_config_name}.train.parquet')
pq.write_table(eval_dataset.data, f'./{data_args.dataset_config_name}.eval.parquet')

# save processor for training
processor.save_pretrained(training_args.output_dir)
