# BCS-Q-Classifier

A Python tool to automatically classify Bangladesh Civil Service (BCS) preliminary exam questions into predefined categories using AI.

## Features

- Classifies BCS MCQs into 10 subject categories
- Uses DeepSeek-V3 model for accurate classification
- Processes questions in chunks with automatic retries
- Saves progress periodically to prevent data loss
- Provides detailed classification statistics

## Categories

- Bengali
- English
- Bangladesh Affairs
- International Affairs
- General Science & Tech
- Computer & IT
- Math Reasoning
- Mental Ability
- Ethics & Good Governance
- Geography

## Requirements

- Python 3.x
- Required packages listed in requirements.txt

## Setup

1. Clone the repository
2. Install dependencies:
```sh
pip install -r requirements.txt
```
3. Create a `.env` file with your Nebius API key:
```
NEBIUS_API_KEY=your_api_key_here
```

## Usage

1. Place your input CSV file containing BCS questions in the project directory
2. Run the classifier:
```sh
python main.py
```

The script will:
- Process questions in batches of 25
- Save intermediate results every 5 chunks
- Generate a final CSV with classifications
- Display classification statistics upon completion

## Input Format

The input CSV should contain a "question" column with BCS MCQs.

## Output

The script generates:
- `bcs_with_categories.csv`: Final output with classified questions
- `bcs_with_categories.csv.partial`: Intermediate saves during processing

## Configuration

Key parameters can be adjusted in `main.py`:
- `CHUNK_SIZE`: Number of questions processed per batch (default: 25)
- `MAX_RETRIES`: Number of API retry attempts (default: 3)
- `RETRY_DELAY`: Seconds between retries (default: 2)