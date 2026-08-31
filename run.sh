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
    stress)
        # Run all stress levels
        echo "Running paraphrase stress tests (none / mild / aggressive)..."

        python3 scripts/paraphrase_stress.py --level none --output results_stress_none_1.json
        echo "Level none done -> results_stress_none_1.json"

        python3 scripts/paraphrase_stress.py --level mild --output results_stress_mild_1.json
        echo "Level mild done -> results_stress_mild_1.json"

        python3 scripts/paraphrase_stress.py --level aggressive --output results_stress_aggressive_1.json
        echo "Level aggressive done -> results_stress_aggressive_1.json"

        echo ""
        echo "=== Stress test summary ==="
        for f in results_stress_none_1.json results_stress_mild_1.json results_stress_aggressive_1.json; do
            echo -n "$f: "
            python3 -c "import json; d=json.load(open('$f')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        done
        ;;
    stress-none)
        echo "Running stress test: none (baseline)..."
        python3 scripts/paraphrase_stress.py --level none --output results_stress_none_1.json
        echo "Done -> results_stress_none_1.json"
        python3 -c "import json; d=json.load(open('results_stress_none_1.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    stress-mild)
        echo "Running stress test: mild (reworded wrappers)..."
        python3 scripts/paraphrase_stress.py --level mild --output results_stress_mild_1.json
        echo "Done -> results_stress_mild_1.json"
        python3 -c "import json; d=json.load(open('results_stress_mild_1.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    stress-aggressive)
        echo "Running stress test: aggressive (markers dropped, synonyms)..."
        python3 scripts/paraphrase_stress.py --level aggressive --output results_stress_aggressive_1.json
        echo "Done -> results_stress_aggressive_1.json"
        python3 -c "import json; d=json.load(open('results_stress_aggressive_1.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    # ========================
    # HYBRID MODE stress tests
    # ========================
    hybrid-stress)
        echo "Running HYBRID paraphrase stress tests (none / mild / aggressive)..."
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level none --output results_hybrid_stress_none.json
        echo "Level none done -> results_hybrid_stress_none.json"
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level mild --output results_hybrid_stress_mild.json
        echo "Level mild done -> results_hybrid_stress_mild.json"
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level aggressive --output results_hybrid_stress_aggressive.json
        echo "Level aggressive done -> results_hybrid_stress_aggressive.json"
        echo ""
        echo "=== HYBRID Stress test summary ==="
        for f in results_hybrid_stress_none.json results_hybrid_stress_mild.json results_hybrid_stress_aggressive.json; do
            echo -n "$f: "
            python3 -c "import json; d=json.load(open('$f')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        done
        ;;
    hybrid-stress-none)
        echo "Running HYBRID stress test: none (baseline)..."
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level none --output results_hybrid_stress_none.json
        echo "Done -> results_hybrid_stress_none.json"
        python3 -c "import json; d=json.load(open('results_hybrid_stress_none.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    hybrid-stress-mild)
        echo "Running HYBRID stress test: mild (reworded wrappers)..."
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level mild --output results_hybrid_stress_mild.json
        echo "Done -> results_hybrid_stress_mild.json"
        python3 -c "import json; d=json.load(open('results_hybrid_stress_mild.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    hybrid-stress-aggressive)
        echo "Running HYBRID stress test: aggressive (markers dropped, synonyms)..."
        RETRIEVAL_MODE=hybrid python3 scripts/paraphrase_stress.py --level aggressive --output results_hybrid_stress_aggressive.json
        echo "Done -> results_hybrid_stress_aggressive.json"
        python3 -c "import json; d=json.load(open('results_hybrid_stress_aggressive.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
    # ========================
    # HYBRID MODE eval
    # ========================
    hybrid-eval)
        echo "Running HYBRID eval..."
        RETRIEVAL_MODE=hybrid python3 -m evaluator.local_evaluator --output results_hybrid.json
        python3 -c "import json; d=json.load(open('results_hybrid.json')); print(f\"score={d.get('recommended_technical_score','?'):.4f}  hit={d.get('hit_rate_at_10','?')}  mrr={d.get('mrr','?')}  mttc={d.get('mttc','?')}\")"
        ;;
esac
