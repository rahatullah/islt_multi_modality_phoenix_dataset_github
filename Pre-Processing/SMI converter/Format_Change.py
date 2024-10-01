import os
import csv

class SMIConverter:
    def __init__(self, smi_path, output_path):
        self.smi_path = smi_path
        self.output_path = output_path

    def convert_smi_to_text(self):
        # Convert all SMI files to timed text format
        for smi_file in os.listdir(self.smi_path):
            smi_data = self.load_smi(os.path.join(self.smi_path, smi_file))
            timed_text = self.convert_to_timed_text(smi_data)
            self.save_as_text(smi_file, timed_text)

    def load_smi(self, smi_file):
        # Load SMI format file (assuming it's CSV or similar)
        with open(smi_file, 'r') as f:
            reader = csv.reader(f)
            smi_data = list(reader)
        return smi_data

    def convert_to_timed_text(self, smi_data):
        # Conversion logic from SMI to timed text
        # For simplicity, returning smi_data as timed text
        timed_text = []
        for row in smi_data:
            start_time, end_time, text = row
            timed_text.append(f"{start_time} --> {end_time}\n{text}\n")
        return timed_text

    def save_as_text(self, smi_file, timed_text):
        # Save timed text to output file
        output_file = os.path.join(self.output_path, smi_file.replace(".smi", ".txt"))
        with open(output_file, 'w') as f:
            f.writelines(timed_text)

if __name__ == "__main__":
    smi_path = "./data/smi_files"
    output_path = "./data/timed_text_files"
    converter = SMIConverter(smi_path, output_path)
    converter.convert_smi_to_text()
