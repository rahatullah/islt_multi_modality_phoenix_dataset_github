import os
import re
from html import unescape
from datetime import timedelta


def process_smi_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    smi_files = [f for f in os.listdir(input_dir) if f.endswith('.smi')]

    for smi_file_name in smi_files:
        input_path = os.path.join(input_dir, smi_file_name)
        output_path = os.path.join(output_dir, smi_file_name.replace('.smi', '.txt'))

        try:
            process_single_smi_file(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


def process_single_smi_file(input_path, output_path):
    with open(input_path, 'rb') as smi_file:
        smi_content = smi_file.read().decode('utf-8')

    timed_plain_text = smi_file_to_timed_plain_text(smi_content)

    if timed_plain_text.strip():
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(timed_plain_text)
        print(f"Processed: {input_path} -> {output_path}")
    else:
        print(f"Skipping: {input_path} (empty output)")


def smi_file_to_timed_plain_text(smi_content):
    pattern = re.compile(r'<SYNC Start=(\d+)><P class=\'en-IN\'>(.*?)\n', re.DOTALL)
    matches = pattern.findall(smi_content)

    decoded_matches = [(timedelta(milliseconds=int(timestamp)), unescape(text)) for timestamp, text in matches]

    formatted_matches = [f"{int(timestamp.total_seconds() // 60):02}:{int(timestamp.total_seconds() % 60):02}\n{text}" for timestamp, text in decoded_matches if text.strip()]

    timed_plain_text = '\n'.join(formatted_matches)
    return timed_plain_text





if __name__ == "__main__":
    input_directory = 'path/to/input/directory'
    output_directory = 'path/to/output/directory'

    process_smi_files(input_directory, output_directory)