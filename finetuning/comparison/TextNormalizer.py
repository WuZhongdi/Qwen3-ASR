try:
    import re2 as re
except ImportError:
    import re

def get_char_map_table():
    # char_table = ['-', ' ', '.', '?', '!', '\'', '\"']
    char_table = ["@-@-@-@"]
    for i in range(26):
        char_table.append(chr(i + 97))
    for i in range(10):
        char_table.append(str(i))
    char_table.append(" ")
    char_table.append(".")
    char_table.append(",")
    char_table.append("\'")
    char_table.append("-")
    return {k: v for k, v in enumerate(char_table)}

class TextNormalizer:
    def __init__(self):
        char_map_table = get_char_map_table()
        allowed_chars = []
        allowed_punctuations = []

        for idx, val in char_map_table.items():
            if idx == 36:  # segmentation symbol
                continue
            if idx == 37:  # space
                continue
            if idx <= 35:  # letters and digits
                allowed_chars.append(val)
            elif idx >= 38:  # punctuation
                allowed_punctuations.append(val)

        self.allowed_chars_pattern = "".join(allowed_chars)
        self.allowed_punctuation_pattern = re.escape("".join(allowed_punctuations))
        self.pattern_remove_brackets = re.compile(r'\[.*?\]|\(.*?\)')
        self.pattern_fillers = re.compile(r'\b(?:hmm|mm|mhm|mmm|uh|um)\b')
        self.pattern_remove_disallowed = re.compile(
            rf"[^{self.allowed_chars_pattern}\s{self.allowed_punctuation_pattern}]"
        )
        self.pattern_space_punctuation = re.compile(rf"\s*([{self.allowed_punctuation_pattern}])\s*")
        self.pattern_multi_periods = re.compile(r"\.{2,}")
        self.pattern_extra_spaces = re.compile(r"\s+")

    def normalize(self, sentence):
        # 1. Lowercase
        text = sentence.lower()

        # 2. Remove anything inside [] or ()
        text = self.pattern_remove_brackets.sub('', text)

        # 3. Remove filler words
        text = self.pattern_fillers.sub('', text)

        # 4. Remove disallowed characters
        text = self.pattern_remove_disallowed.sub('', text)

        # 5. Remove spaces around punctuation
        text = self.pattern_space_punctuation.sub(r"\1", text)

        # 6. remove multiple periods
        text = self.pattern_multi_periods.sub(' ', text)

        # 7. Remove extra spaces
        text = self.pattern_extra_spaces.sub(' ', text).strip()

        # 8. Ensure ends with a single period
        if not text.endswith('.'):
            text += '.'

        return text
