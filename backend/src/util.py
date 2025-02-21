import nltk
import string
import warnings

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.exceptions import UndefinedMetricWarning

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UndefinedMetricWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

STOP_WORDS = set(stopwords.words("english"))


def preprocess(input: str):
    """Basic text preprocessing on input text
    Args:
        input (str): Text to be preprocessed
    Returns:
        _type_: Prerpocessed string
    """

    if not isinstance(input, str):
        return ""

    # Text to lower case
    input = input.lower()

    # Remove punctuation.
    input = input.translate(str.maketrans("", "", string.punctuation))

    # Tokenize input
    t = word_tokenize(input)

    # Remove stopwords
    t = [w for w in t if w not in STOP_WORDS]

    # Lemmatize input
    l = WordNetLemmatizer()

    return " ".join([l.lemmatize(w) for w in t])
