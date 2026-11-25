from torch.utils.data.dataset import Dataset
import numpy as np
import random
import torch


class MLMLoader(Dataset):
    def __init__(self, dataframe, token2idx, age2idx, max_len, sex_dict=None, code='code', age='age'):
        self.max_len = max_len
        self.vocab_size = len(token2idx)

        self.cls_id  = token2idx['CLS']
        self.sep_id  = token2idx['SEP']
        self.pad_id  = token2idx['PAD']
        self.mask_id = token2idx['MASK']
        self.unk_id  = token2idx.get('UNK', 0)

        self.age_pad_id = age2idx.get('PAD', 0)

        unk = self.unk_id

        # The corpus may arrive ALREADY ENCODED as int32 id arrays: holding the
        # full MarketScan corpus as Python strings needs ~132 GB against ~24 GB
        # encoded (see read_sequences.read_encoded).  Accept either form.
        #
        # Without this branch, integer input fails SILENTLY rather than raising:
        # token2idx is keyed by strings, so token2idx.get(<int>, unk) misses on
        # every token and yields UNK, and the model pre-trains on a corpus of
        # nothing but unknown tokens.
        _first = dataframe[code].iloc[0] if len(dataframe) else []
        _pre_encoded = (hasattr(_first, 'dtype')
                        and np.issubdtype(_first.dtype, np.integer))

        if _pre_encoded:
            self.code_ids = list(dataframe[code])
            self.age_ids  = list(dataframe[age])
        else:
            self.code_ids = [
                np.array([token2idx.get(c, unk) for c in row], dtype=np.int32)
                for row in dataframe[code]
            ]
            self.age_ids = [
                np.array([age2idx.get(str(a), 0) for a in row], dtype=np.int32)
                for row in dataframe[age]
            ]

        # Gate: any corpus/vocabulary mismatch shows up as a flood of UNK.
        # Sampled, because the corpus has billions of tokens.
        if self.code_ids:
            _step = max(1, len(self.code_ids) // 20_000)
            _sample = np.concatenate(self.code_ids[::_step])
            if _sample.size:
                _unk_frac = float((_sample == unk).mean())
                if _unk_frac > 0.02:
                    raise ValueError(
                        f'{_unk_frac:.1%} of sampled tokens are UNK, which means '
                        f'the corpus and the vocabulary do not match. Check that '
                        f'token2idx is the one the corpus was encoded with. '
                        f'(pre_encoded={_pre_encoded}, vocab={len(token2idx):,})')
                self.unk_fraction = _unk_frac

        # Store sex values (static per patient, will be broadcast to all positions)
        if sex_dict is not None and 'enrolid' in dataframe.columns:
            self.sex_values = np.array([
                sex_dict.get(enrolid, 0) for enrolid in dataframe['enrolid']
            ], dtype=np.int32)
        else:
            self.sex_values = np.zeros(len(dataframe), dtype=np.int32)

    def __getitem__(self, index):
        """return: age, code, position, segmentation, mask, label, sex"""
        ml = self.max_len

        code = self.code_ids[index][-(ml - 1):]
        age  = self.age_ids[index][-(ml - 1):]

        if code[0] != self.sep_id:
            code = np.concatenate(([self.cls_id], code))
            age  = np.concatenate(([age[0]], age))
        else:
            code = code.copy()
            code[0] = self.cls_id

        seq_len = len(code)

        mask = np.ones(ml, dtype=np.int32)
        mask[seq_len:] = 0

        age_padded = np.full(ml, self.age_pad_id, dtype=np.int32)
        age_padded[:seq_len] = age[:seq_len]

        # Broadcast sex to all positions (static feature)
        sex_broadcast = np.full(ml, self.sex_values[index], dtype=np.int32)

        out_tokens = np.full(ml, self.pad_id, dtype=np.int32)
        out_labels = np.full(ml, -100, dtype=np.int32)

        for i in range(seq_len):
            cid = int(code[i])
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    out_tokens[i] = self.mask_id
                elif prob < 0.9:
                    out_tokens[i] = random.randint(0, self.vocab_size - 1)
                else:
                    out_tokens[i] = cid
                out_labels[i] = cid
            else:
                out_tokens[i] = cid
                out_labels[i] = -100

        position = np.zeros(ml, dtype=np.int32)
        segment  = np.zeros(ml, dtype=np.int32)
        pos_counter = 0
        seg_flag = 0
        for i in range(ml):
            tok = out_tokens[i]
            position[i] = pos_counter
            segment[i]  = seg_flag
            if tok == self.sep_id:
                pos_counter += 1
                seg_flag = 1 - seg_flag

        return (torch.LongTensor(age_padded),
                torch.LongTensor(out_tokens),
                torch.LongTensor(position),
                torch.LongTensor(segment),
                torch.LongTensor(mask),
                torch.LongTensor(out_labels),
                torch.LongTensor(sex_broadcast))

    def __len__(self):
        return len(self.code_ids)
