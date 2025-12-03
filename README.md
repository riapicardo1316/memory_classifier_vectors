# Memory Classifier

This script reads a conversation text file and classifies each sentence into memory categories such as health, identity, medication, appointments, goals, and others. It performs TTL decay, contradiction handling, repetition detection, and produces final short-term and long-term memory states.

## Requirements

Create a file named `requirements.txt` with the following lines:

```
transformers
sentence-transformers
scikit-learn
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Install Python 3.8 or above.
2. Clone or place the project folder locally.
3. Create a text file containing user conversation. Each line should be one statement. Example:

```
conversation.txt
```

4. Ensure the text file is in the same directory as the script.

## Running the Script

Run the classifier by passing the conversation file name as an argument:

```bash
python3 memory_classifier.py conversation.txt
```

If the file does not exist, the script prints an error and stops.
