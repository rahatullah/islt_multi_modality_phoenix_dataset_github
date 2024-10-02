import gzip
import pandas as pd

def convert_gzip_to_excel(gzip_path, excel_path):
    with gzip.open(gzip_path, 'rt', encoding='utf-8') as f:
        data = pd.read_csv(f, delimiter='|')  # Adjust the delimiter as per the dataset
        data.to_excel(excel_path, index=False)
    print(f"Converted {gzip_path} to {excel_path}")

# Convert all .gzip files
if __name__ == "__main__":
    convert_gzip_to_excel('Dataset/phoenix14t.pami0.train.annotations_only.gzip', 'Dataset/excels/Train.xlsx')
    convert_gzip_to_excel('Dataset/phoenix14t.pami0.dev.annotations_only.gzip', 'Dataset/excels/Dev.xlsx')
    convert_gzip_to_excel('Dataset/phoenix14t.pami0.test.annotations_only.gzip', 'Dataset/excels/Test.xlsx')
