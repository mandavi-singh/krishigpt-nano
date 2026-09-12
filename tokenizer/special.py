"""Special tokens shared by all tokenizers in this project.

WHY SPECIAL TOKENS EXIST
------------------------
The model only ever sees integers. Some integers must carry *structural*
meaning that no real text character can carry:

  <PAD> (id 0)  PADDING. Batches need equal-length rows, so short sequences
                are padded with this id. Attention/loss masks later ignore it
                (ignore_index=-100 in the loss, or explicit masks).
  <UNK> (id 1)  UNKNOWN. Anything the tokenizer cannot represent (a character
                or word never seen during vocabulary construction) maps here
                instead of crashing.
  <BOS> (id 2)  BEGINNING-OF-SEQUENCE. Marks where a sequence starts, so the
                model can condition "this is a fresh sequence". Useful for
                generation and for chat/structured formats.
  <EOS> (id 3)  END-OF-SEQUENCE. During generation, when the model samples
                <EOS> we STOP. Also separates documents/turns in training data.

They occupy fixed ids 0..3. Real text tokens start at id 4.
We deliberately keep only these four: extra special tokens add complexity
without teaching anything new at this stage.
"""
from __future__ import annotations

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
NUM_RESERVED_IDS: int = len(SPECIAL_TOKENS)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_IDS = frozenset(range(NUM_RESERVED_IDS))


def is_special_id(token_id: int) -> bool:
    """True if `token_id` is one of the reserved special tokens."""
    return token_id in SPECIAL_IDS
