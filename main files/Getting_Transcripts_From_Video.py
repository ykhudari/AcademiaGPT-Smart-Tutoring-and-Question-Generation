import os
import moviepy.editor as mp
import whisper
from whisper.utils import get_writer

def transcribe_videos(transcripts_tiny):
    if not os.path.exists(transcripts_tiny):
        print(f"Directory '{transcripts_tiny}' not found.")
        return
    
    # Loading the whisper model
    model = whisper.load_model("tiny")
    
    # Creating a directory to store transcripts if it doesn't exist
    output_directory = os.path.join(directory, 'transcripts_tiny')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Processing all mp4 files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory, filename)
            output_audio_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.mp3")
            
            # Extracting audio from the video
            clip = mp.VideoFileClip(video_path)
            audio_file = clip.audio
            audio_file.write_audiofile(output_audio_path)
            
            # Transcribing the audio
            result = model.transcribe(output_audio_path)
            
            # Creating SRT file for transcript
            srt_output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.srt")
            options = {
                'max_line_width': None,
                'max_line_count': None,
                'highlight_words': False
            }
            srt_writer = get_writer("srt", output_directory)
            srt_writer(result, output_audio_path, options)
            
            print(f"Transcription complete for {filename}.")
    
    print("Transcription process completed for all videos.")


directory_path = 'transcripts_tiny'
transcribe_videos(directory_path)
