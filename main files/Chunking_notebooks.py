import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_md_files(directory_path):
    # Ensuring the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
    
    # Iterating through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            input_file_path = os.path.join(directory_path, filename)
            with open(input_file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()
            
            # Splitting text using RecursiveCharacterTextSplitter
            ss = RecursiveCharacterTextSplitter(chunk_size=1000)
            split_content = ss.split_text(markdown_content)
            
            # Creating a directory to store the split content if it doesn't exist
            output_directory = os.path.join(directory_path, 'split_md_files')
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            
            # Writing split content to separate files
            for i, chunk in enumerate(split_content):
                output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_part_{i+1}.md")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(chunk)
            
            print(f"Splitting complete for {filename}.")
    
    print("Splitting process completed for all Markdown files.")

directory_path = 'jupyteach-ai/AcademiaGPT Project/Parsed Notebooks/'
split_md_files(directory_path)
