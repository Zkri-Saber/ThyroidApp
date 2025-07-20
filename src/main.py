from thyroid_analysis.pipeline import run_pipeline
from thyroid_analysis.config import OUTPUT_DIR
import os

def main():
    # ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = run_pipeline()
    print("Pipeline results:\n", results)

    out_path = OUTPUT_DIR / "pipeline_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
