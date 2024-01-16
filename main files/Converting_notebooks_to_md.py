import nbformat
import os

def convert_notebooks_to_markdown(directory_path):
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return
    
    # Iterating through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(directory_path, filename)
            with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
                notebook_content = nbformat.read(notebook_file, as_version=4)
            
            # Function to convert cells to markdown
            def cells_to_markdown(cells):
                markdown_output = ""
                for cell in cells:
                    if cell.cell_type == 'markdown':
                        markdown_output += cell.source + '\n\n'
                    elif cell.cell_type == 'code':
                        markdown_output += f'```python\n{cell.source}\n```\n\n'
                return markdown_output
            
            markdown_content = cells_to_markdown(notebook_content.cells)
            
            file_name = os.path.splitext(filename)[0]
            md_file_name = f"{file_name}.md"
            
            with open(md_file_name, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            
            print(f"Conversion complete for {filename}. Generated {md_file_name}.")
    
    print("Conversion process completed for all notebooks.")

# Replace 'directory_path' with the directory containing your .ipynb files
directory_path = 'notebooks/All_notebooks'
convert_notebooks_to_markdown(directory_path)
