import unittest
from pathlib import Path

from dads_crnn.prepare_g13_ddl import contiguous_chunks


class DDLPreparationTests(unittest.TestCase):
    def test_chunks_do_not_cross_sequence_gap(self) -> None:
        items = [(value, Path(f"{value}.wav")) for value in list(range(20)) + list(range(30, 45))]
        chunks = contiguous_chunks(items)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([item[0] for item in chunks[-1]], list(range(30, 40)))


if __name__ == "__main__":
    unittest.main()
