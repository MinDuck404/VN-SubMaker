import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer
from pydub import AudioSegment
import moviepy.editor as mp
import matplotlib.pyplot as plt

# Đường dẫn đến mô hình của bạn và mô hình Whisper truyền thống
YOUR_MODEL_PATH = "ASR"  # Thay bằng đường dẫn mô hình của bạn
WHISPER_MODEL_PATH = "openai/whisper-base"  # Mô hình Whisper base từ Hugging Face

# Thiết bị và kiểu dữ liệu
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Tải mô hình của bạn
your_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    YOUR_MODEL_PATH, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
your_model.to(device)
your_processor = AutoProcessor.from_pretrained(YOUR_MODEL_PATH)
your_pipe = pipeline(
    "automatic-speech-recognition",
    model=your_model,
    tokenizer=your_processor.tokenizer,
    feature_extractor=your_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "vi"}
)

# Tải mô hình Whisper truyền thống
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL_PATH,
    device=device,
    generate_kwargs={"language": "vi"}
)

# Hàm chuyển đổi file âm thanh sang định dạng WAV
def convert_to_wav(input_path, output_path):
    if input_path.lower().endswith('.mp4'):
        video = mp.VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(output_path, logger=None)
        video.close()
    else:
        AudioSegment.from_file(input_path).export(output_path, format="wav")
    return output_path

# Hàm chia nhỏ file âm thanh
def split_audio(audio_path, chunk_length=25):
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i:i + chunk_length * 1000] for i in range(0, len(audio), chunk_length * 1000)]
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

# Hàm tính WER và CER
def compute_metrics(reference_text, hypothesis_text):
    word_error_rate = wer(reference_text, hypothesis_text)
    char_error_rate = cer(reference_text, hypothesis_text)
    return word_error_rate, char_error_rate

# Hàm đánh giá trên tập dữ liệu
def evaluate_models(test_data_dir, max_files=500):
    your_results = []
    whisper_results = []
    audio_files = [f for f in os.listdir(test_data_dir) if f.endswith(('.wav', '.mp3', '.mp4'))][:max_files]
    
    for idx, audio_file in enumerate(audio_files):
        print(f"Processing {idx+1}/{len(audio_files)}: {audio_file}")
        audio_path = os.path.join(test_data_dir, audio_file)
        ref_text_file = os.path.join(test_data_dir, audio_file.rsplit('.', 1)[0] + '.txt')
        
        try:
            with open(ref_text_file, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
        except FileNotFoundError:
            print(f"Skipping {audio_file}: Reference text file not found.")
            continue
        
        wav_path = "temp.wav"
        convert_to_wav(audio_path, wav_path)
        chunk_paths = split_audio(wav_path)
        
        # Nhận dạng với mô hình của bạn
        your_text = ""
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                your_result = your_pipe(chunk_path, return_timestamps=False)
                your_text += your_result['text'].strip() + " "
                os.remove(chunk_path)
        your_text = your_text.strip()
        
        # Nhận dạng với Whisper
        chunk_paths = split_audio(wav_path)
        whisper_text = ""
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                whisper_result = whisper_pipe(chunk_path, return_timestamps=False)
                whisper_text += whisper_result['text'].strip() + " "
                os.remove(chunk_path)
        whisper_text = whisper_text.strip()
        
        # Tính WER và CER
        your_wer, your_cer = compute_metrics(reference_text, your_text)
        whisper_wer, whisper_cer = compute_metrics(reference_text, whisper_text)
        
        your_results.append((audio_file, your_wer, your_cer))
        whisper_results.append((audio_file, whisper_wer, whisper_cer))
        
        if os.path.exists(wav_path):
            os.remove(wav_path)
    
    return your_results, whisper_results

# Hàm vẽ biểu đồ
def plot_comparison(your_results, whisper_results):
    # Tính trung bình
    total_your_wer = sum(r[1] for r in your_results) / len(your_results) if your_results else 0
    total_your_cer = sum(r[2] for r in your_results) / len(your_results) if your_results else 0
    total_whisper_wer = sum(r[1] for r in whisper_results) / len(whisper_results) if whisper_results else 0
    total_whisper_cer = sum(r[2] for r in whisper_results) / len(whisper_results) if whisper_results else 0

    # 1. Biểu đồ cột so sánh trung bình
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['ASR Model', 'Whisper']
    wer_values = [total_your_wer, total_whisper_wer]
    cer_values = [total_your_cer, total_whisper_cer]
    
    ax[0].bar(models, wer_values, color=['blue', 'orange'])
    ax[0].set_title('Average WER Comparison')
    ax[0].set_ylabel('WER')
    for i, v in enumerate(wer_values):
        ax[0].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    ax[1].bar(models, cer_values, color=['blue', 'orange'])
    ax[1].set_title('Average CER Comparison')
    ax[1].set_ylabel('CER')
    for i, v in enumerate(cer_values):
        ax[1].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()

    # 2. Biểu đồ phân tán so sánh WER từng file
    plt.figure(figsize=(10, 6))
    file_indices = range(len(your_results))
    your_wers = [r[1] for r in your_results]
    whisper_wers = [r[1] for r in whisper_results]
    
    plt.scatter(file_indices, your_wers, color='blue', label='ASR Model WER', alpha=0.5)
    plt.scatter(file_indices, whisper_wers, color='orange', label='Whisper WER', alpha=0.5)
    plt.title('WER Comparison per File')
    plt.xlabel('File Index')
    plt.ylabel('WER')
    plt.legend()
    plt.grid(True)
    plt.show()

# Đường dẫn đến thư mục chứa dữ liệu kiểm tra
TEST_DATA_DIR = r"C:\Users\duccj\Downloads\Compressed\vlsp2020_train_set_02"

if __name__ == "__main__":
    print(f"Device set to use {device}")
    # Đánh giá mô hình với 500 file
    your_results, whisper_results = evaluate_models(TEST_DATA_DIR, max_files=500)
    
    # Vẽ biểu đồ so sánh
    plot_comparison(your_results, whisper_results)