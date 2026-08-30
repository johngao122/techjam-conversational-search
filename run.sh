#!/bin/bash
set -e

case "${1:-eval}" in
    try)
        # Interactive parser with LLM-based conversation summarization.
        # Requires Ollama running locally: ollama serve (in another terminal)
        # Uses local phi3:mini model for summaries (no API rate limits)
        #
        # Setup (one-time):
        #   ollama pull phi3:mini  
        #   pip install ollama
        #
        # Run:
        #   ./run.sh try
        python3 -m src.message_parser.try_it
        ;;
    eval)
        python3 -m evaluator.local_evaluator --output results_ours.json
        ;;
esac
