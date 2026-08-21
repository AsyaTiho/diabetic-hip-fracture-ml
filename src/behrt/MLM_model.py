import torch.nn as nn
import numpy as np
import torch
from transformers import BertConfig as HFBertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)


class BehrtConfig(HFBertConfig):
    def __init__(self, seg_vocab_size=2, age_vocab_size=200, sex_vocab_size=3, **kwargs):
        super().__init__(**kwargs)
        self.seg_vocab_size = seg_vocab_size
        self.age_vocab_size = age_vocab_size
        self.sex_vocab_size = sex_vocab_size  # 0=PAD, 1=male, 2=female


class BehrtEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age, position, and sex."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.sex_embeddings = nn.Embedding(config.sex_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.posi_embeddings = self.posi_embeddings.from_pretrained(
            embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size)
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, sex_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)
        if sex_ids is None:
            sex_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        sex_embed = self.sex_embeddings(sex_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + sex_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + sex_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BehrtModel(BertPreTrainedModel):
    config_class = BehrtConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BehrtEmbeddings(config=config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.post_init()

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, sex_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)
        if sex_ids is None:
            sex_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Use first parameter dtype when available, otherwise fall back to float32
        _params = list(self.parameters())
        _mask_dtype = _params[0].dtype if _params else torch.float32
        extended_attention_mask = extended_attention_mask.to(dtype=_mask_dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids, sex_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BehrtForMaskedLM(BertPreTrainedModel):
    config_class = BehrtConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BehrtModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, sex_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, age_ids, seg_ids, posi_ids, sex_ids, attention_mask)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
        else:
            return prediction_scores
