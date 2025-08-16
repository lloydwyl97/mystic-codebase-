from strategy_prompt_builder import generate_strategy_code


def create_strategy_from_prompt(prompt, file_name):
    code = generate_strategy_code(prompt)
    with open(f"strategies/{file_name}.py", "w") as f:
        f.write(code)
    print(f"[LLM] Created {file_name}.py")


