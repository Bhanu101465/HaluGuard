# Part 5: inside signal (EASY version).
# Ask the model "Is this answer true? True/False" and read how sure it is
# from the token probabilities. If the provider has no logprobs -> 0.5.
import math

def signal_score(question: str, answer: str) -> float:
    try:
        from .llm import _get_client
        client, model = _get_client()
        prompt = (
            "Question: " + question + "\nProposed answer: " + answer +
            "\n\nIs the proposed answer correct? Reply with only: True or False."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        p_true = p_false = 0.0
        for t in top:
            word = t.token.strip().lower()
            if word == "true":
                p_true += math.exp(t.logprob)
            elif word == "false":
                p_false += math.exp(t.logprob)
        if p_true + p_false == 0:
            return 0.5
        return p_true / (p_true + p_false)
    except Exception:
        return 0.5  # neutral: no logprobs or call failed

if __name__ == "__main__":
    # Fake client test (no real model)
    import types, app.signals as me
    def mk(tok, lp):
        return types.SimpleNamespace(token=tok, logprob=lp)
    top = [mk("True", math.log(0.9)), mk("False", math.log(0.1))]
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(
        logprobs=types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)]))])
    fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **k: resp)))
    import app.llm as llm
    llm._get_client = lambda: (fake_client, "m")
    print(round(signal_score("q", "a"), 2))   # 0.9
