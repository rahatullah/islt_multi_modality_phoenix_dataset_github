import torch
from torch.utils.data import DataLoader

class SignBatch:
    """
    Batching logic for multi-stream input: visual, emotion, and gesture.
    """

    def __init__(self, visual_input, emotion_input, gesture_input, target_sentences):
        self.visual_input = visual_input
        self.emotion_input = emotion_input
        self.gesture_input = gesture_input
        self.target_sentences = target_sentences

    @staticmethod
    def make_batch(data_list):
        # Create batches for each input stream
        visual_input = [item[0] for item in data_list]
        emotion_input = [item[1] for item in data_list]
        gesture_input = [item[2] for item in data_list]
        target_sentences = [item[3] for item in data_list]

        return SignBatch(
            visual_input=torch.stack(visual_input),
            emotion_input=torch.stack(emotion_input),
            gesture_input=torch.stack(gesture_input),
            target_sentences=torch.stack(target_sentences)
        )

def load_data(data_path, annotations_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the multi-stream dataset.
    :param data_path: path to video data
    :param annotations_path: path to annotations
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :return: DataLoader
    """
    dataset = SignDataset(data_path=data_path, annotations_path=annotations_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=SignBatch.make_batch)
