# assistant/main.py

from core.pipeline import Pipeline

def main():
    print("Assistant 시작 (종료: exit)")

    pipeline = Pipeline()

    while True:
        user_input = input("\n👤 사용자 명령: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        output = pipeline.run(user_input)
        print(f"🤖 Assistant: {output}")


if __name__ == "__main__":
    main()