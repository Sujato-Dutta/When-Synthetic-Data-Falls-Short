"""Fix all emoji/non-ASCII characters in print statements across all project Python files."""
import os, glob

TARGET_DIR = os.path.join(os.path.dirname(__file__), "synthetic_hate")

REPLACEMENTS = [
    ('\U0001f3af', '[TARGET]'),
    ('\u2705', '[OK]'),
    ('\u26a0\ufe0f', '[WARN]'),
    ('\u26a0', '[WARN]'),
    ('\U0001f4c2', '[FOLDER]'),
    ('\u23ed', '[SKIP]'),
    ('\u25b6', '[RUN]'),
    ('\U0001f389', '[DONE]'),
    ('\u2192', '->'),
    ('\U0001f3c6', '[BEST]'),
    ('\u2714', '[OK]'),
    ('\u274c', '[X]'),
    ('\U0001f4be', '[SAVE]'),
    ('\u231b', '[WAIT]'),
    ('\u23f1', '[TIME]'),
]

files = glob.glob(os.path.join(TARGET_DIR, '**', '*.py'), recursive=True)
total = 0
for fpath in files:
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    new = content
    for emoji, rep in REPLACEMENTS:
        new = new.replace(emoji, rep)
    if new != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new)
        print('Fixed:', os.path.basename(fpath))
        total += 1

print('Done. Fixed', total, 'files.')
