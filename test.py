import assemblyai as aai

aai.settings.api_key = "32bede1c60e44fb59a506e389067972d"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("123.mp4")

subtitle = transcript.export_subtitles_srt()

f = open("123.srt", "a")
f.write(subtitle)
f.close()
