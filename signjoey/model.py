import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, TransformerEncoder
from signjoey.decoders import TransformerDecoder
from signjoey.vocabulary import GlossVocabulary, TextVocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from torch import Tensor
from typing import Union


class SignModel(nn.Module):
    """
    Upgraded model class to handle multi-stream input (visual, emotion, gesture) while
    maintaining original gloss recognition and translation capabilities.
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: TransformerDecoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

        # New components for visual, emotion, and gesture streams
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.mediapipe_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mediapipe_hands = mp.solutions.hands.Hands()

    def forward(
        self,
        visual_input: Tensor,
        emotion_input: Tensor,
        gesture_input: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        """
        Forward pass for multi-stream input: visual, emotion, and gesture.
        """

        # EfficientNet for visual feature extraction
        visual_features = self.efficientnet(visual_input)
        visual_encoded = self.encoder(embed_src=self.sgn_embed(visual_features))

        # MediaPipe for emotion and gesture feature extraction
        emotion_features = self.extract_mediapipe_features(emotion_input, self.mediapipe_face_mesh)
        emotion_encoded = self.encoder(embed_src=self.sgn_embed(emotion_features))

        gesture_features = self.extract_mediapipe_features(gesture_input, self.mediapipe_hands)
        gesture_encoded = self.encoder(embed_src=self.sgn_embed(gesture_features))

        # Concatenate the encoded outputs from visual, emotion, and gesture streams
        encoder_output = torch.cat((visual_encoded, emotion_encoded, gesture_encoded), dim=-1)

        if self.do_recognition:
            # Gloss recognition
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2).permute(1, 0, 2)  # T x N x C
        else:
            gloss_probabilities = None

        if self.do_translation:
            # Translation decoding
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def extract_mediapipe_features(self, input_data, mediapipe_solution):
        """
        Extract MediaPipe landmarks for emotion/gesture streams.
        """
        results = mediapipe_solution.process(input_data)
        # Process the landmarks into a suitable tensor format for the model
        return torch.tensor(...)  # Process to extract tensor format

    def decode(
        self,
        encoder_output: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        """
        Decode the concatenated input streams into text output.
        """
        return self.decoder(
            encoder_output=encoder_output,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute the loss for recognition and translation tasks for a batch.
        """
        # Forward pass
        decoder_outputs, gloss_probabilities = self.forward(
            visual_input=batch.visual_input,
            emotion_input=batch.emotion_input,
            gesture_input=batch.gesture_input,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        # Recognition loss (CTC loss)
        if self.do_recognition:
            assert gloss_probabilities is not None
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        # Translation loss
        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = translation_loss_function(txt_log_probs, batch.txt) * translation_loss_weight
        else:
            translation_loss = None

        return recognition_loss, translation_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attention scores for a given batch.
        """
        encoder_output, encoder_hidden = self.encode(
            visual_input=batch.visual_input,
            emotion_input=batch.emotion_input,
            gesture_input=batch.gesture_input,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
        )

        if self.do_recognition:
            # Gloss recognition using beam search or greedy decoding
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2).permute(1, 0, 2).cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )
            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            decoded_gloss_sequences = self._convert_ctc_decode_to_sequence(ctc_decode, gloss_scores.shape[0])
        else:
            decoded_gloss_sequences = None

        # Translation decoding
        if self.do_translation:
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
            else:
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output, stacked_attention_scores = None, None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def encode(
        self, visual_input: Tensor, emotion_input: Tensor, gesture_input: Tensor, sgn_mask: Tensor, sgn_lengths: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the concatenated visual, emotion, and gesture inputs.
        """
        visual_features = self.efficientnet(visual_input)
        visual_encoded = self.encoder(embed_src=self.sgn_embed(visual_features))

        emotion_features = self.extract_mediapipe_features(emotion_input, self.mediapipe_face_mesh)
        emotion_encoded = self.encoder(embed_src=self.sgn_embed(emotion_features))

        gesture_features = self.extract_mediapipe_features(gesture_input, self.mediapipe_hands)
        gesture_encoded = self.encoder(embed_src=self.sgn_embed(gesture_features))

        # Concatenate encoded features
        encoder_output = torch.cat((visual_encoded, emotion_encoded, gesture_encoded), dim=-1)

        return encoder_output, None  # Return encoder output and hidden states

    def _convert_ctc_decode_to_sequence(self, ctc_decode, batch_size):
        """
        Helper function to convert CTC beam search decode results to gloss sequences.
        """
        tmp_gloss_sequences = [[] for _ in range(batch_size)]
        for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
            tmp_gloss_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 1)

        decoded_gloss_sequences = []
        for seq_idx in range(len(tmp_gloss_sequences)):
            decoded_gloss_sequences.append([x[0] for x in groupby(tmp_gloss_sequences[seq_idx])])

        return decoded_gloss_sequences

    def __repr__(self) -> str:
        """
        String representation of the model architecture.
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the upgraded multi-stream model according to the configuration.
    """
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # Build encoder
    encoder = TransformerEncoder(
        **cfg["encoder"],
        emb_size=sgn_embed.embedding_dim,
        emb_dropout=cfg["encoder"]["embeddings"].get("dropout", 0.1),
    )

    # Gloss output layer for recognition
    gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))

    # Build decoder
    txt_embed = Embeddings(
        **cfg["decoder"]["embeddings"],
        num_heads=cfg["decoder"]["num_heads"],
        vocab_size=len(txt_vocab),
        padding_idx=txt_padding_idx,
    )
    decoder = TransformerDecoder(
        **cfg["decoder"],
        encoder=encoder,
        vocab_size=len(txt_vocab),
        emb_size=txt_embed.embedding_dim,
        emb_dropout=cfg["decoder"]["embeddings"].get("dropout", 0.1),
    )

    model = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    initialize_model(model, cfg, txt_padding_idx)
    return model
