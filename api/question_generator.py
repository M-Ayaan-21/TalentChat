import openai

def generate_questions(candidate_name, candidate_level, candidate_embeddings):
    """
    Generate 10 AI-powered questions for a candidate:
    - 5 aptitude multiple-choice
    - 2 coding
    - 3 scenario-based
    Returns a list of dicts: [{"question": "...", "type": "..."}, ...]
    """
    # Prepare a concise embedding summary for the prompt
    embedding_preview = candidate_embeddings[:5] if candidate_embeddings else []

    prompt = f"""
You are an AI talent acquisition assistant.
Candidate Name: {candidate_name}
Level: {candidate_level}
Resume Embeddings (truncated): {embedding_preview}

Generate:
- 5 aptitude multiple-choice questions
- 2 coding questions
- 3 scenario-based questions

Return the result as a JSON array with each item:
{{"question": "<question text>", "type": "<aptitude|coding|scenario>"}}
    """

    # Call OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.7
    )

    # Extract content
    content = response['choices'][0]['message']['content']

    # Try to parse as JSON
    try:
        import json
        questions = json.loads(content)
    except Exception:
        # Fallback if JSON parsing fails
        questions = [{"question": content, "type": "mixed"}]

    return questions

