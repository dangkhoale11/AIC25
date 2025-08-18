import unittest
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from core.translation import TextTranslator

class TestTextTranslator(unittest.TestCase):

    def setUp(self):
        self.translator = TextTranslator()

    def test_translate_vietnamese_to_english(self):
        vietnamese_text = "Chào thế giới"
        translated_text = self.translator.translate(vietnamese_text)
        self.assertEqual(translated_text.lower(), "hello world")

    def test_translate_english_to_english(self):
        english_text = "Hello world"
        translated_text = self.translator.translate(english_text)
        self.assertEqual(translated_text, "Hello world")

    def test_translate_empty_string(self):
        empty_text = ""
        translated_text = self.translator.translate(empty_text)
        self.assertEqual(translated_text, "")

    def test_translate_string_with_spaces(self):
        space_text = "   "
        translated_text = self.translator.translate(space_text)
        self.assertEqual(translated_text, "")

if __name__ == '__main__':
    unittest.main()
