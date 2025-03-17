---
datasets:
- capleaf/viVoice
- NhutP/VSV-1100
- doof-ferb/fpt_fosd
- doof-ferb/infore1_25hours
- google/fleurs
- doof-ferb/LSVSC
- quocanh34/viet_vlsp
- linhtran92/viet_youtube_asr_corpus_v2
- doof-ferb/infore2_audiobooks
- linhtran92/viet_bud500
language:
- vi
metrics:
- wer
base_model:
- openai/whisper-large-v3-turbo
new_version: suzii/vi-whisper-large-v3-turbo
library_name: transformers
---
# Fine-tuned Whisper-V3-Turbo for Vietnamese ASR

This project involves fine-tuning the Whisper-V3-Turbo model to improve its performance for Automatic Speech Recognition (ASR) in the Vietnamese language. The model was trained for 240 hours using a single Nvidia A6000 GPU.

## Data Sources

The training data comes from various Vietnamese speech corpora. Below is a list of datasets used for training:

1. **capleaf/viVoice**  
2. **NhutP/VSV-1100**  
3. **doof-ferb/fpt_fosd**  
4. **doof-ferb/infore1_25hours**  
5. **google/fleurs (vi_vn)**  
6. **doof-ferb/LSVSC**  
7. **quocanh34/viet_vlsp**  
8. **linhtran92/viet_youtube_asr_corpus_v2**  
9. **doof-ferb/infore2_audiobooks**  
10. **linhtran92/viet_bud500**

## Model

The model used in this project is the **Whisper-V3-Turbo**. Whisper is a multilingual ASR model trained on a large and diverse dataset. The version used here has been fine-tuned specifically for the Vietnamese language.

## Training Configuration

- **GPU Used**: Nvidia A6000  
- **Training Time**: 240 hours
- [Wandb report](https://api.wandb.ai/links/goiliace/ae0qectc)




## Usage

To use the fine-tuned model, follow the steps below:

  ```python
  import torch
  from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
  
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
  
  model_id = "suzii/vi-whisper-large-v3-turbo-v1"
  
  model = AutoModelForSpeechSeq2Seq.from_pretrained(
      model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
  )
  model.to(device)
  
  processor = AutoProcessor.from_pretrained(model_id)
  
  pipe = pipeline(
      "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      torch_dtype=torch_dtype,
      device=device,
  )
  result = pipe("your-audio.mp3", return_timestamps=True)

  ```

## Acknowledgements

This project would not be possible without the following datasets:

- [capleaf/viVoice](https://huggingface.co/datasets/capleaf/viVoice)
- [NhutP/VSV-1100](https://huggingface.co/datasets/nhutp/vsv-1100)
- [doof-ferb/fpt_fosd](https://huggingface.co/datasets/doof-ferb/fpt_fosd)
- [doof-ferb/infore1_25hours](https://huggingface.co/datasets/doof-ferb/infore1_25hours)
- [google/fleurs](https://huggingface.co/datasets/google/fleurs)
- [doof-ferb/LSVSC](https://huggingface.co/datasets/doof-ferb/LSVSC)
- [quocanh34/viet_vlsp](https://huggingface.co/datasets/quocanh34/viet-vlsp)
- [linhtran92/viet_youtube_asr_corpus_v2](https://huggingface.co/datasets/linhtran92/viet_youtube_asr_corpus_v2)
- [doof-ferb/infore2_audiobooks](https://huggingface.co/datasets/doof-ferb/infore2_audiobooks/)
- [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500)