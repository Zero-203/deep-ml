import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    res = re.split(r'([,.:;?_!"()\']|--|\s)',text)
    res = [x for x in res if x!='' and x!=' ' and x!='\t' and x!='\n']
    return res