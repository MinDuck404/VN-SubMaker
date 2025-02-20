from flask import Flask, render_template, request, jsonify, Response, send_file
import os
import queue
import threading
import json
import time
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import moviepy.editor as mp
from datetime import timedelta
import threading

# Global dictionary to store transcription results
transcription_results = {}
transcription_locks = {}

app = Flask(__name__)

# Thay model_path bằng đường dẫn của bạn
MODEL_PATH = "vi-whisper-large-v3-turbo-v1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

status_queues = {}

def convert_to_wav(input_path, wav_path):
    if input_path.lower().endswith('.mp4'):
        video = mp.VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(wav_path)
        video.close()
    else:
        AudioSegment.from_file(input_path).export(wav_path, format="wav")
    return wav_path

def generate_srt(result):
    """
    Tạo file SRT từ kết quả nhận dạng giọng nói.
    """
    chunks = result.get('chunks', [])
    srt_content = []
    
    for i, chunk in enumerate(chunks, 1):
        start_time = max(0, chunk['timestamp'][0] or 0)
        end_time = max(start_time + 1, chunk['timestamp'][1] or (start_time + 1))

        # Format timestamps đúng chuẩn SRT (HH:MM:SS,mmm)
        start_timecode = format_srt_timestamp(start_time)
        end_timecode = format_srt_timestamp(end_time)

        text = chunk['text'].strip()

        if text:
            srt_content.append(f"{i}\n{start_timecode} --> {end_timecode}\n{text}\n")

    return "\n".join(srt_content)

def format_srt_timestamp(seconds):
    """
    Chuyển đổi thời gian (tính bằng giây) sang định dạng SRT (HH:MM:SS,mmm).
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def split_audio_into_chunks(audio_path, chunk_length=4):
    """
    Chia file âm thanh thành từng đoạn nhỏ có độ dài chunk_length (giây).
    """
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i * 1000 * chunk_length:(i + 1) * 1000 * chunk_length] 
              for i in range(len(audio) // (chunk_length * 1000) + 1)]
    
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"{audio_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def process_transcription(session_id, audio_path):
    try:
        queue = status_queues[session_id]
        queue.put({"status": "processing", "message": "Đang xử lý file âm thanh..."})

        # Chuyển đổi video thành audio WAV
        wav_path = audio_path.replace(os.path.splitext(audio_path)[1], '.wav')
        convert_to_wav(audio_path, wav_path)

        # Chia audio thành các đoạn 4 giây
        queue.put({"status": "processing", "message": "Đang chia đoạn audio..."})
        chunk_paths = split_audio_into_chunks(wav_path, chunk_length=4)

        queue.put({"status": "processing", "message": "Đang nhận dạng từng đoạn..."})
        all_chunks = []
        start_time = 0

        for i, chunk_path in enumerate(chunk_paths):
            result = pipe(chunk_path, return_timestamps=True)
            text = result.get('text', '').strip()
            if text:
                all_chunks.append({
                    "timestamp": (start_time, start_time + 4),
                    "text": text
                })
            start_time += 4
            os.remove(chunk_path)

        # Ghép các đoạn nhận dạng vào SRT
        srt_result = generate_srt({"chunks": all_chunks})

        # Lưu kết quả vào global dictionary
        with threading.Lock():
            transcription_results[session_id] = {
                "text": "\n".join([c["text"] for c in all_chunks]),
                "srt": srt_result
            }

        queue.put({
            "status": "completed",
            "message": "Hoàn thành nhận dạng!",
            "text": transcription_results[session_id]["text"],
            "srt": srt_result
        })

        os.remove(audio_path)
        os.remove(wav_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        queue.put({"status": "error", "message": str(e)})

def generate_srt_improved (result):
    """
    Generate an advanced SRT subtitle file with robust timestamp and text handling.
    
    Args:
        result (dict): Whisper transcription result dictionary
    
    Returns:
        str: Formatted SRT subtitle content
    """
    chunks = result.get('chunks', [])
    srt_content = []
    
    for i, chunk in enumerate(chunks, 1):
        # Robust timestamp handling with fallbacks
        timestamps = chunk.get('timestamp', [None, None])
        start_time = max(0, float(timestamps[0] or 0))
        end_time = max(start_time + 1, float(timestamps[1] or (start_time + 1)))
        
        # Format timestamps to SRT timecode format
        start_timecode = _format_timecode(start_time)
        end_timecode = _format_timecode(end_time)
        
        # Clean and prepare chunk text
        text = chunk.get('text', '').strip()
        
        # Only add non-empty chunks
        if text:
            srt_content.append(f"{i}\n{start_timecode} --> {end_timecode}\n{text}\n")
    
    return "\n".join(srt_content)

def _format_timecode(seconds):
    """
    Convert seconds to SRT timecode format (HH:MM:SS,mmm)
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timecode
    """
    td = timedelta(seconds=max(0, seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Include milliseconds with 3 decimal digits
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
@app.route('/')
def index():
    return render_template('2index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'status': 'error', 'message': 'Missing audio file'}), 400

        session_id = str(time.time())
        status_queues[session_id] = queue.Queue()

        os.makedirs('uploads', exist_ok=True)
        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)

        threading.Thread(
            target=process_transcription,
            args=(session_id, audio_path)
        ).start()

        return jsonify({'status': 'started', 'session_id': session_id})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status/<session_id>')
def status_stream(session_id):
    def generate():
        if session_id not in status_queues:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid session'})}\n\n"
            return

        queue = status_queues[session_id]
        while True:
            try:
                data = queue.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                if data['status'] in ['completed', 'error']:
                    break
            except queue.Empty:
                break

        if session_id in status_queues:
            del status_queues[session_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download_text/<session_id>')
def download_text(session_id):
    result = transcription_results.get(session_id)
    if result:
        return Response(
            result['text'],
            mimetype='text/plain',
            headers={'Content-Disposition': 'attachment; filename=transcription.txt'}
        )
    return "No completed transcription found", 404

@app.route('/download_srt/<session_id>')
def download_srt(session_id):
    result = transcription_results.get(session_id)
    if result:
        srt_content = result['srt']

        # Lưu file với encoding UTF-8 không BOM
        srt_path = f"uploads/{session_id}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        return send_file(
            srt_path,
            mimetype="text/plain",
            as_attachment=True,
            download_name="subtitles.srt"
        )
    return "No completed transcription found", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)