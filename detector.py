import argparse
from utils.feature_based import feature_ai_score
from utils.model_based import model_ai_score
from utils.hybrid import hybrid_predict_with_calibrator, load_calibrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="분석할 텍스트 파일")
    parser.add_argument("--text", type=str, help="직접 입력할 텍스트")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("텍스트를 입력하세요 (--file 또는 --text)")
        return

    feature_score = feature_ai_score(text)
    model_score = model_ai_score(text)
    calibrator = load_calibrator()
    final_score = hybrid_predict_with_calibrator(feature_score, model_score, calibrator)
    print(f"\nAI 작성 가능성: {final_score:.2f}%")

if __name__ == "__main__":
    main()
