import re
from nltk.tokenize import sent_tokenize
import html
import random
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags=re.UNICODE)

def camel_terms(value):
    return re.findall('[A-Z][a-z]+|[0-9A-Z]+(?=[A-Z][a-z])|[0-9A-Z]{2,}|[a-z0-9]{2,}|[a-zA-Z0-9]', value)

def construct_irl_text(irl, const_action=False):
    # TODO: add ablation study logic (different IRL subsets)

    org_utterance = irl['text']
    frames = irl['frames']
    if len(frames) == 0:
        return org_utterance, False

    valid_spans = []
    for frame in frames:
        spans = frame['spans']
        # Heuristic 0: Set STOPWORDS
        for span in spans:
            text = span['text']
            if text in stops:
                span['label'] = 'Stopword'

        # IRL Labels: Action, Argument, Request, Query, Slot, Problem
        labels = set([span['label'] for span in spans])
        labels -= set(['Slot']) # to filter
        if len(labels) == 0:
            continue

        valid_spans += [{'text': span['text'], 'start': span['start']} for span in spans]
    
    if len(valid_spans) == 0:
        return org_utterance, False
  
    valid_spans = sorted(valid_spans, key = lambda x: x['start'])
    irl_text = ' '.join([i['text'] for i in valid_spans])
    return irl_text, True

def filter_text(text, single_sent=False):
    # start filtering criteria
    if len(text.split()) < 3:       # text is shorter than 3 words
        return False                    # (also filters out [removed] or [deleted] comments)
    if len(text.split()) > 20:     # text is paragraph-length or longer. probably not intentful utterance.
        return False
    if "http://" in text or "https://" in text or ".com " in text:   # contains URL. these utterances tend to be messy/noisy
        return False
    if single_sent and len(sent_tokenize(text)) >1:
        return False
    return True
                        
def process_text(text):
    text = emoji_pattern.sub(r'', text) # remove emoji
    text = text.replace("\n", ' ')
    text = text.replace("|", ' ')
    text = html.unescape(text)
    
    return text

def sample_min_num(intent2utterances, min_num):
    new_intent2utterances = {}
    for intent,utterances in intent2utterances.items():
        utterances = list(set(utterances))
        random.shuffle(utterances)
        new_intent2utterances[intent] = utterances[:min_num]
    intent2utterances = new_intent2utterances

    final_intents = []
    final_utterances = []
    for intent, utterances in intent2utterances.items():
        for utt in utterances:
            final_utterances.append(utt)
            final_intents.append(intent)
    
    intent2counts = {intent:len(utterances) for intent,utterances in intent2utterances.items()}
    intent2counts = dict(sorted(intent2counts.items(), key=lambda item: item[1], reverse=True))
    
    # print(intent2counts)
    print("len(intents):",len(intent2counts))
    print("len(utterances):",len(final_utterances))

    assert len(final_utterances) == len(final_intents)
    return final_utterances, final_intents
