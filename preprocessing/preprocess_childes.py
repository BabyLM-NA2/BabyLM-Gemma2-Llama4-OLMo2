import glob
import pandas as pd
import re
import random
import tqdm
import os 

# DOCUMENTATION: """https://talkbank.org/manuals/CHAT.pdf"""


INCLUDE_TAGS = [
    "%act", # Action?
    "%flo", # this always follows `www .`. TODO: Add a special condition for this?
    "%exp", # Experimenter's commentary?
    "%par", # Paralinguistic?
    "%com", # Commentary?
    "%gpx", # Gesture?
    "%sit", # Situation?
    "%int", # Intonation?
    "%add", # Addressee
]


def get_record(filename):
    text = ""
    metadata = ""
    participants = {}
    for l in open(filename):
        if l.startswith("@Situation"):
            text += l
        if l.startswith("@Participants"):
            participants.update({kv.split()[0].strip(): kv.split()[1] for kv in l.split("\t")[1].split(",")})
        elif l.startswith("@"):
            metadata += l
        elif l.startswith("*"):
            text += l
        elif l.startswith("%"):
            if l.split("\t")[0][:-1] in INCLUDE_TAGS:
                text += l
        elif not l.startswith("\t"):
            print(l)
    record = {
        "filename": filename,
        "text": text,
        "metadata": metadata,
        "participants": participants
    }
    return record


def merge_consecutive_speakers(text):
    # Merge content from consecutive speakers
    lines = text.split('\n')
    merged = []
    current_speaker = None
    current_content = []
    
    for line in lines:
        # Match speaker labels
        speaker_match = re.match(r'^(\*\w+:\t)(.+)', line)
        if speaker_match:
            speaker_tag, content = speaker_match.groups()
            # Check if it is the same speaker
            if speaker_tag == current_speaker:
                current_content.append(content)
            else:
                # Write the previous speaker's content
                if current_speaker:
                    merged.append(f"{current_speaker}{' '.join(current_content)}")
                current_speaker = speaker_tag
                current_content = [content]
        else:
            # Non-speech lines are directly reserved
            merged.append(line)
    
    # Handling the last speaker
    if current_speaker:
        merged.append(f"{current_speaker}{' '.join(current_content)}")
    
    return '\n'.join(merged)


DEBUG = False
INCLUDE_SPKR = True
def process_text(textstring):
    textstring = re.sub(r".*", "", textstring)
    original = textstring

    # NON SPEECH ROWS
    if DEBUG:
        textstring = re.sub(r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<bef> (.*?)(\n|^)", r"\1[\4]\2\5\2", textstring)
        textstring = re.sub(r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<aft> (.*?)(\n|^)", r"\1\2[\4]\5\2", textstring)
    else:
        textstring = re.sub(r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<bef> (.*?)(\n|^)", r"\1[\4]\2\5", textstring)
        textstring = re.sub(r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<aft> (.*?)(\n|^)", r"\1\2[\4]\5", textstring)
    textstring = re.sub(r"%(act|exp|par|com|gpx|sit):\t(.*?)(\n|^)", r"[\2]\3", textstring)
    textstring = re.sub(r"@Situation:\t(.*?)(\n|^)", r"[\1]\2", textstring)


    # GOING THROUGH CHILDES ANNOTATIONS
    ## Some things just need to be done up top
    ### 10.5 errors
    textstring = re.sub(r"\[\*[^\]]+?\]", "", textstring)  # error

    ### Dealing with colons
    #### 10.3 Explanations and alternatives
    textstring = re.sub(r"\[:+.*?\]", "", textstring)  # replacements transcriptions get deleted
    #### 9.10 Local events
    textstring = re.sub(r"&=0[\w:]+", "", textstring)   # this removes things like "sneezes" which someone might want, but we'll exclude it
    textstring = re.sub(r"&=(\S+)", r"[\1]", textstring)
    for _ in range(5):
        textstring = re.sub(r"\[(\S+)[:_](.*?)\]", r"[\1 \2]", textstring)

    ## Going through in order
    ### 8.3 Special form markers
    textstring = re.sub(r"@[\w:\$\*]+", "", textstring)

    ### 8.5 Fragments, Fillers, and Nonwords
    textstring = re.sub(r"&\+[\w]+", "", textstring)
    textstring = re.sub(r"&\-", "", textstring)
    textstring = re.sub(r"&\~", "", textstring)

    ### 8.6 Incomplete and Omitted Words
    textstring = re.sub(r"\(\w+?\)", "'", textstring)
    textstring = re.sub(r"0\w*", "", textstring)

    ### 8.8 Standardized Spellings
    textstring = re.sub(r"@l", "", textstring)
    textstring = re.sub(r"(\w)\_", r"\1 ", textstring)

    ### 9.3 Satellite Markers
    textstring = re.sub(r"[‡„]", "", textstring)

    ### 9.8 Tone Direction
    textstring = re.sub(r"[↑↓]", "", textstring)

    ### 9.9 Prosody Within Words
    textstring = re.sub(r"[ˈˌ≠](\w)", r"\1", textstring)
    textstring = re.sub(r"(\w)[:\^]", r"\1", textstring)

    ### 9.10 Local events
    textstring = re.sub(r"&\*\w+:\w+", "", textstring)   # this removes interposed words
    textstring = re.sub(r"\[\^.*?\]", "", textstring)   # this removes text descriptions of actions (rare)
    textstring = re.sub(r"\(\d*\.+\d*\)", "", textstring)   # this removes pauses
    textstring = re.sub(r"&\{.*&\}", "", textstring)

    ### 9.11 Utterance terminators
    textstring = re.sub(r"\+\.\.\.", "...", textstring)
    textstring = re.sub(r"\+\.\.\?", "...?", textstring)
    textstring = re.sub(r"\+!\?", "!?", textstring)
    textstring = re.sub(r"\+/\.", "...", textstring)
    textstring = re.sub(r"\+/\?", "...?", textstring)
    textstring = re.sub(r"\+//\.", "...", textstring)
    textstring = re.sub(r"\+//\?", "...?", textstring)
    textstring = re.sub(r"\+\.", "", textstring)
    textstring = re.sub(r"\+\"/\.", "", textstring)
    textstring = re.sub(r"\+\"\.", "", textstring)

    ### 9.12 Utterance linkers
    textstring = re.sub(r"\+\"(.*?)($|\n)", r'"\1"\2', textstring)
    textstring = re.sub(r"\+(\^|,|\+)", "", textstring)

    ### 10.2 Paralinguistic & duration scoping
    textstring = re.sub(r"\[=!\s?(.*)\]", r"[\1]", textstring)  # paralinguistic material is saved
    textstring = re.sub(r"\[!+\]", "", textstring)  # stress gets deleted
    textstring = re.sub(r"\[#.*\]", "", textstring)  # duration gets deleted
    textstring = re.sub(r"\[:.*\]", "", textstring)  # corrections get deleted

    ### 10.3 Explanations and alternatives
    textstring = re.sub(r"\[=\s?(.*?)\]", r"[\1]", textstring)  # explanations get saved
    textstring = re.sub(r"\[=\?.*?\]", "", textstring)  # alternative transcriptions get deleted
    textstring = re.sub(r"\[%\s?(.*?)\]", r"[\1]", textstring)  # comments get saved
    textstring = re.sub(r"\[\?\]", "", textstring)  # best guess gets deleted

    ### 10.4 Retracing, Overlap, Exclusions, and Clauses
    textstring = re.sub(r"\[[<>]\d?\]", "", textstring)  # overlap gets deleted
    textstring = re.sub(r"\+<", "", textstring)  # lazy overlap gets deleted
    textstring = re.sub(r"(<.+?>|\S+) \[/\]", "", textstring)  # repetition gets deleted. This might be controversial
    textstring = re.sub(r"(<.+?>|\S+) \[//\]", "", textstring)  # repetition gets deleted. This might be controversial
    textstring = re.sub(r"\[/\]", "", textstring)  # If there's no angle brackets, I can't tell what was repeated, so I keep the repetition but delete the annot.
    textstring = re.sub(r"\[//\]", "", textstring)  # If there's no angle brackets, I can't tell what was repeated, so I keep the repetition but delete the annot.
    textstring = re.sub(r"(<(.+?)>|\S+) \[///\]", r"\1 ...", textstring)  # reformulation: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(r"(<(.+?)>|\S+) \[/-\]", r"\1 ...", textstring)  # false start: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(r"(<(.+?)>|\S+) \[/\?\]", r"", textstring)  # unclear retracing
    textstring = re.sub(r"(<(.+?)>|\S+) \[(e|\+ exc)\]", r"\1 ...", textstring)  # reformulation: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(r"\[\^c.*?\]", "", textstring)  # clause delimiter

    ### 10.5 errors
    textstring = re.sub(r"\[\*\]", "", textstring)  # error

    ### 10.6 Precodes and Postcodes
    textstring = re.sub(r"( \[\+ \w+\])+\s*[\"\s]?($|\n)", r"\2", textstring)  # postcode
    textstring = re.sub(r"\t(\[\- \w +\] )+", "", textstring)  # precode

    ### Extra delimiters
    textstring = re.sub(r"[<>]", "", textstring)


    # CLEANUP
    ## fix spaces and puncutation
    textstring = re.sub(r"  +", " ", textstring)
    textstring = re.sub(r"\t +", "\t", textstring)
    textstring = re.sub(r" +([\.,\?!])", r"\1", textstring)
    textstring = re.sub(r"(^|\n)([\*%]\w+)", r"\1\2:", textstring)

    ## remove empty lines
    textstring = re.sub(r"(\*\w+:\t)(xxx|yyy|www|0|\.)\s?[\.\?]? ?(\[.*\])\s?[\.\?]? ?($|\n)", r"\1\3\4", textstring)  # some empty utterance followed by an action
    if DEBUG:
        textstring = re.sub(r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", r"____\2", textstring)
    else:
        textstring = re.sub(r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", "", textstring)

    ## remove speaker?
    # if not INCLUDE_SPKR:
    textstring = re.sub(r"\*\w+:\t", "", textstring)

    ## remove repeat lines
    lines = textstring.split("\n")
    textstring = ""
    l_prev = ""
    for l in lines:
        if l == l_prev:
            continue
        textstring += l + "\n"
        l_prev = l

    # Convert to lowercase
    textstring = textstring.lower()

    # Remove all punctuation/special characters except letters, numbers, spaces, and square brackets
    textstring = re.sub(r"[^\w\s\[\]'\-]+", "", textstring)

    # Merge consecutive speeches
    textstring = merge_consecutive_speakers(textstring)

    # Remove bracketed insertions or annotations inside utterances
    textstring = re.sub(r"\[[^\[\]]*?\]", "", textstring)

    # Remove any leftover unmatched [ or ] characters
    textstring = re.sub(r"[\[\]]", "", textstring)

    ## random cleanup
    textstring = re.sub(r"\[= :\d+ \]", "", textstring)

    if DEBUG:
        return "\n".join(a + "\n" + b for a, b in zip(textstring.split("\n"), original.split("\n")))
    else:
        return textstring


def incorporate_metadata(text, record):
    # for k, v in record["participants"].items():
    #     text = re.sub(r"\*?"+k, v, text)
    header = "= = = " + record["filename"] + " = = ="
    text = header + "\n" + text
    return text


if __name__ == "__main__":
    data_folder = os.getenv('DATA_FOLDER')
    input_file = f'data/{data_folder}/childes.train' if data_folder != 'dev' else f'data/{data_folder}/childes.dev'
    output_dir = f'data/{data_folder}_cleaned'
    os.makedirs(output_dir, exist_ok=True)  # Automatically create output directories
    output_file = os.path.join(output_dir, 'childes.train') if data_folder != 'dev' else os.path.join(output_dir, 'childes.dev')

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip().startswith('%'):  # Skip %int:, %com:, etc.
                continue
            if not line.strip():  # Skip originally blank lines
                continue

            cleaned = process_text(line).strip()
            
            if cleaned:
                outfile.write(cleaned + '\n')

    print(f"Finished preprocessing. Saved to {output_file}")