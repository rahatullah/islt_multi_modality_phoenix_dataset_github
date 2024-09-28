import pickle
import gzip
import os
pickle_dest = os.getcwd() +"/Dataset/Pickles/excel_data1.test"
# pickle_dest = "/Dataset/Pickles/excel_dataVal.pickle"
# pickle_dest = "/Dataset/Pickles/excel_dataTrain.pickle"
def view_gzip_pickle_file(file_path):
    try:
        with gzip.open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the gzip-compressed pickle file:")
            print(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
view_gzip_pickle_file(pickle_dest)
