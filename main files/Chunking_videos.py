import os
import re

rex = re.compile(r"\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}")

def chunk_srt_files(full_text, chunk_length):
    splits = rex.split(full_text)[1:]

    # Combining parts into a list of 3-tuples (start, end, txt)
    parts = []
    for i in range(0, len(splits), 3):
        start_time = splits[i]
        end_time = splits[i+1]
        content = splits[i+2].strip()
        parts.append((start_time, end_time, content))

    # Combining multiple parts to get desired chunk length
    chunks = []
    ix = 0
    current_chunk_text = ""
    for i, part in enumerate(parts):
        current_chunk_text = current_chunk_text + " " + part[2]
        if len(current_chunk_text) > chunk_length or i == len(parts) - 1:
            current_chunk = (
                parts[ix][0],  # starting timestamp
                part[1],
                current_chunk_text.strip()
            )
            chunks.append(current_chunk)
            ix = i  # Repeat this chunk one more time for overlap
            current_chunk_text =  part[2]

    return chunks

def process_all_srt_files(input_folder_srt, output_folder_chunk, chunk_length):
    if not os.path.exists(output_folder_chunk):
        os.makedirs(output_folder_chunk)

    # Listing all SRT files in the input folder
    srt_files = [f for f in os.listdir(input_folder_srt) if f.endswith('.srt')]

    for srt_file in srt_files:
        # Read the content of the SRT file
        with open(os.path.join(input_folder_srt, srt_file)) as f:
            txt = f.read()

        # Calling the chunk_srt_files function to process the SRT content
        chunks = chunk_srt_files(txt, chunk_length)

        # Saving the chunked content to the output folder
        output_file = os.path.join(output_folder_chunk, srt_file)
        with open(output_file, 'w') as f:
            for chunk in chunks:
                f.write(f"{chunk[0]} --> {chunk[1]}\n{chunk[2]}\n\n")

# Setting the paths for the input and output folders and specify the chunk length
input_folder_srt = "/videos/transcripts_tiny"
output_folder_chunk = "/videos/transcripts_chunked"
chunk_length = 1000

# Calling the function to process all SRT files in the input folder
process_all_srt_files(input_folder_srt, output_folder_chunk, chunk_length)
