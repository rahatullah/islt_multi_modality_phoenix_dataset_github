import os
import pickle
import cv2
import mediapipe as mp

class PickleCreator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.hands = mp.solutions.hands.Hands()

    def process_videos(self):
        # Process all videos in the given path
        for video_file in os.listdir(self.video_path):
            video_frames = self.load_video(os.path.join(self.video_path, video_file))

            # Extract visual, emotion, and gesture features
            visual_features = self.extract_visual_features(video_frames)
            emotion_features = self.extract_mediapipe_features(video_frames, self.face_mesh)
            gesture_features = self.extract_mediapipe_features(video_frames, self.hands)

            # Save features in pickle format
            self.save_as_pickle(video_file, visual_features, emotion_features, gesture_features)

    def load_video(self, video_path):
        # Load video and convert to frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_visual_features(self, frames):
        # Placeholder for EfficientNet visual extraction, for now returning raw frames
        return frames  # Ideally, EfficientNet will process this

    def extract_mediapipe_features(self, frames, mediapipe_solution):
        features = []
        for frame in frames:
            result = mediapipe_solution.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks or result.multi_hand_landmarks:
                landmarks = result.multi_face_landmarks if mediapipe_solution == self.face_mesh else result.multi_hand_landmarks
                for lm in landmarks:
                    features.append(lm)
            else:
                features.append([0] * 128)  # Default zero features if no landmarks found
        return features

    def save_as_pickle(self, video_file, visual_features, emotion_features, gesture_features):
        # Save features into a pickle file
        pickle_data = {
            'visual': visual_features,
            'emotion': emotion_features,
            'gesture': gesture_features
        }
        output_file = os.path.join(self.output_path, video_file.replace(".mp4", ".pkl"))
        with open(output_file, 'wb') as f:
            pickle.dump(pickle_data, f)

if __name__ == "__main__":
    video_path = "./data/videos_train"
    output_path = "./data/pickle_data_train"
    creator = PickleCreator(video_path, output_path)
    creator.process_videos()
