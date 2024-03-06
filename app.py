import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
import torch.nn.functional as F

from functools import partial

VOCAB_SIZE = 6
LSTM_HIDDEN = 32
LSTM_LAYER = 4


class CpGPredictor(torch.nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(VOCAB_SIZE, LSTM_HIDDEN, LSTM_LAYER, batch_first=True)
        self.classifier = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        packed_output, _ = self.lstm(x)
        encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        encoded_sum = torch.sum(encoded, dim=1)
        logits = self.classifier(encoded_sum).squeeze(-1)
        return logits


def preprocess_sequence(dna_sequence):
    alphabet = 'NACGT'
    dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}

    dnaseq_to_intseq = partial(map, dna2int.get)

    sequences = torch.stack([torch.tensor(list(dnaseq_to_intseq(dna_sequence)))])
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    sequences = F.one_hot(sequences, num_classes=6).to(torch.float32)
    packed_sequences = pack_padded_sequence(sequences, lengths, batch_first=True)

    return packed_sequences


model = CpGPredictor()
model.load_state_dict(torch.load("autonomize_lstm.pt"))
model.eval()

st.title("DNA Sequence CpG Counter")
dna_sequence = st.text_input("Enter a DNA sequence:", "")

if st.button("Count CpGs"):
    processed_sequence = preprocess_sequence(dna_sequence)
    output = model(processed_sequence).item()
    st.write("Raw LSTM Output:", round(output, 2))
