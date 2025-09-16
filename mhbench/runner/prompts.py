IMHI_CHOICE_PROMPT = """You are a careful mental-health classifier.
Choose exactly one label from the list and return JSON with keys "label" and "confidence" (0.0-1.0).

Labels: {labels}

Text:
{input}

Answer JSON:
"""

IMHI_OPEN_PROMPT = """You are a careful mental-health classifier.
From the following label set, choose exactly one label and return JSON {{ "label": <one_of_labels>, "confidence": <0..1> }}.

Label set: {labels}

Text:
{input}

Answer JSON:
"""

DIALOGUE_PROMPT = """You are an empathetic, supportive assistant.
Respond concisely and compassionately to the seeker.

Seeker: {seeker}
Assistant:"""

COUNSELCHAT_PROMPT = """You are a licensed-therapist style assistant. Provide a supportive, safe, practical response.
Question: {question}
Answer:"""

DSM_MCQ_PROMPT = """You are a psychiatry expert.
Select the single best answer to the following DSM-5 style multiple-choice question.
Return JSON with keys "label" and "confidence" (0.0–1.0).

Question:
{question}

Choices:
{choices}

Answer JSON:
"""

DSM_EXPLANATION_PROMPT = """You are a psychiatry expert.
Given the following question and the correct answer, provide a short explanation
that justifies why this answer is correct. The explanation should be factually accurate
and based on DSM-5 clinical criteria.

Question:
{question}

Correct Answer: {answer}

Explanation:"""

DSM_VIGNETTE_PROMPT = """You are a psychiatry expert.
Read the following clinical vignette carefully and provide the most likely DSM-5 diagnosis,
along with a brief rationale. Return JSON with keys "label" and "rationale".

Vignette:
{vignette}

Answer JSON:
"""
