import ollama
import os
import chardet

mypath = "./files"
files = [
    os.path.join(mypath, f)
    for f in os.listdir(mypath)
    if os.path.isfile(os.path.join(mypath, f))
]
print(files)
context = []
for f in files:
    with open(f, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
        decoded_data = raw_data.decode(encoding)
        new_lines = [line for line in decoded_data.splitlines() if line.startswith("+")]
        context.append("\n".join(new_lines))
print(context)
if files:
    prompt = f"""I will provide txt files that are git diffs. I need to know if there are any new
functions implemented without docstrings to explain code.
Those are the files on a list:
{context}
Only analyze new code. Give me a list of functions without docstring and write me a docstring for them.
"""

    response = ollama.generate(model="phi4", prompt=prompt).response
    print(response)
