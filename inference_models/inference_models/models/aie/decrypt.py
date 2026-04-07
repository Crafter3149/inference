import math
from typing import Optional

import numpy as np


def decrypt_nst(nst_path: str) -> Optional[bytes]:
    """Decrypt an AIE .nst encrypted weight file.

    Args:
        nst_path: Path to the .nst file.

    Returns:
        Decrypted bytes if the header matches, None otherwise.
    """
    with open(nst_path, "rb") as f:
        content = f.read()

    header = [24, 97, 28, 98]
    for i in range(4):
        if int(content[i]) != header[i]:
            return None

    random_num_range = 87
    header_len = 4
    cut_num_start = random_num_range + header_len
    cut_num = int.from_bytes(
        content[cut_num_start : cut_num_start + 4],
        byteorder="little",
        signed=False,
    )
    compliment_num_start = cut_num_start + 4
    compliment_num = int.from_bytes(
        content[compliment_num_start : compliment_num_start + 4],
        byteorder="little",
        signed=False,
    )
    seq_num_start = compliment_num_start + 4
    seq = content[seq_num_start : seq_num_start + cut_num]
    pb_fragment = content[seq_num_start + cut_num :]
    leng = len(pb_fragment)
    slice_num = math.ceil(leng / cut_num)

    seq2num = [int(s) for s in seq]
    sort_seq = np.argsort(seq2num)

    temp = b""
    for num, i in enumerate(sort_seq):
        num_start = slice_num * i
        num_end = num_start + slice_num
        if num == cut_num - 1:
            num_end -= compliment_num
        temp += pb_fragment[num_start:num_end]
    temp = temp[::-1]
    return temp
