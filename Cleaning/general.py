import os
import textwrap


with open('Data/information_smaller.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

single_line_text = raw_text.replace('\n', ' ').replace('\r', ' ')
single_line_text = ' '.join(single_line_text.split())  

chunk_size = 2048
chunks = textwrap.wrap(single_line_text, chunk_size)

output_dir = 'llc_info_chunks'
os.makedirs(output_dir, exist_ok=True)

for idx, chunk in enumerate(chunks):
    with open(os.path.join(output_dir, f'chunk_info_{idx+1}.txt'), 'w', encoding='utf-8') as f:
        f.write(chunk)

print(len(chunks))