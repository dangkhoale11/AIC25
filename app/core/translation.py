from googletrans import Translator, LANGUAGES

class TextTranslator:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text: str, dest: str = 'en') -> str:
        if not text.strip():
            return ""

        try:
            detected = self.translator.detect(text)
            if detected.lang == 'vi':
                return self.translator.translate(text, dest=dest).text
            return text
        except Exception as e:
            print(f"Error during translation: {e}")
            return text
