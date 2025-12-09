from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import librosa 
processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-4017")
model.to("cpu")
speech, sr = librosa.load("Sample.mp3", sr=16000)
inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).logits
lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_language = model.config.id2label[lang_id]
print(f"Detected language: {detected_language}")

