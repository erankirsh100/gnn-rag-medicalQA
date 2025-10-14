import pandas as pd
import re
import google.generativeai as genai
import os
import time
from scipy.stats import wilcoxon
import numpy as np
from itertools import cycle
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import json

RAG_GNN_scores=[]
Vanilla_LLM_scores=[]

# Parse JSON array of API keys
api_keys_str = os.getenv('API_KEYS')
api_keys = json.loads(api_keys_str) if api_keys_str else []

class APIKeyManager:
    """
    Round-robin API key manager with failover retry.
    """
    def __init__(self, keys):
        if not keys:
            raise ValueError("Provide at least one API key.")
        self.keys = keys
        self._cycle = cycle(self.keys)
        self.current_key = None

    def configure_next(self):
        """Rotate to the next key and configure genai."""
        self.current_key = next(self._cycle)
        os.environ["API_KEY"] = self.current_key
        genai.configure(api_key=self.current_key)
        return self.current_key

    def try_with_retry(self, func, *args, **kwargs):
        """
        Call func with failover retry on exception.
        """
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[APIKeyManager] Key {self.current_key} failed → rotating: {e}")
                self.configure_next()
                time.sleep(1)

    def __copy__(self):
        new_manager = APIKeyManager(self.keys)
        new_manager.current_key = self.current_key
        new_manager._cycle = self._cycle
        return new_manager

global api_manager
api_manager = APIKeyManager(api_keys)

def rag_gnn_generation_prompt(symptom_description, retrieved_contexts, gnn_knowledge):
    prompt = f"""### TASK OVERVIEW:
You are an AI-powered clinical decision support assistant. You combine:
- **Free-text patient symptom input**
- **Structured medical knowledge** from a graph-based system (GNN) encoding disease-symptom relations and clinical ontologies
 # GNN may reveal **latent associations** between symptoms and diseases not obvious in text.
 #Use GNN knowledge to spot comorbidities, symptom clusters, or rare conditions that align with the patient's case.
- **Retrieved medical evidence** from pubmed papers (metadata and abstracts) via a RAG (Retrieval-Augmented Generation) pipeline

Your role is to:
- Be **thoughtful, accurate, and empathetic**, and answer **full-text response** that helps explain the user's possible medical condition, supports it with reasoning, and advises a safe and appropriate next step
- Generate a clear, well-reasoned **diagnostic hypothesis**
- Provide **rationale** grounded in the inputs
- Suggest **responsible, evidence-aware next steps**
- Be transparent about **uncertainty and limitations**
- Encourage users to **seek real medical care**
— while making it clear that this is not a substitute for a real physician.

---

### INPUTS:

- **Symptom Description:**
{symptom_description}

- **Retrieved Medical Contexts (from RAG):**
{retrieved_contexts}

- **Structured Graph Knowledge (from GNN):**
{gnn_knowledge}

---

### OUTPUT STRUCTURE:

You must return a **structured but natural-sounding clinical response** that includes the following sections — all written as one continuous narrative:

---

#### 1. DIAGNOSIS HYPOTHESIS
- Clearly identify the most likely primary diagnosis or diagnoses based on the provided symptoms, history, risk factors and our pipeline data (RAG context, and graph-based knowledge). highlight key symptom patterns, onset, duration, and associated findings that support the hypothesis.
- You may suggest more than one but **not to much to avoid confusion**. Do not present a list - instead, **write it as a flowing paragraph**.
- Where applicable, integrate epidemiological clues (age, sex, comorbidities, geographical location, seasonality), and explicitly state how they change the likelihood of a diagnosis (‘This condition is more common in adolescents…’, ‘This presentation is less typical in females…’).
- Consider the temporal relationship of symptoms (e.g., acute vs. chronic onset, progression pattern) as part of the diagnostic reasoning.
- When the presentation is nonspecific, explain which additional details or tests would most help narrow the possibilities.
- Summarize the hypothesis with a clear prioritization: primary likely cause, followed by alternative but less probable causes.
- If there is not enough evidence for a confident diagnosis, clearly state that and provide reasonable hypotheses.

---

#### 2. CLINICAL REASONING

- Explain your reasoning clearly and logically, describing how the symptoms contribute to the diagnostic hypotheses.
- Support your conclusions by referencing specific information from the retrieved medical contexts, citing article titles or study names when relevant and available.
- You may also refer to relevant supporting evidence more generally as “medical sources,” “clinical guidelines,” “published studies,” “medical literature”, etc.
- For example:
  - “Medical literature generally recognizes that symptoms A, B, and C often appear together in cases of X.”
  - “A study titled ‘[Article Title]’ (Author et al., YEAR) found that symptom A commonly presents with symptom B, which supports the possibility of X in this case.”
  - “Clinical guidelines from [Organization Name] (YEAR) recommend considering test Y when symptoms A and B are present.”
  - “It is well established in clinical practice that symptom A can be an early indicator of condition Y.”
  - “The recommended treatment for symptoms A, B, and C is X [direct suporting quotation] ([Author et al.] (YEAR), ‘[Article Title]’)”  (direct quotes are welcome!)

  When available, always include at least one reference to a medical guideline, study, or reputable source that supports your reasoning, integrating it naturally in the text.
- Incorporate structured clinical knowledge naturally by referring to well-known symptom clusters, disease co-occurrences, or common clinical patterns. For example, say:
  - “It is well documented that symptoms A, B, and C frequently occur together in cases of X.”
  - “Clinical experience and published cases commonly show that Y is associated with Z.”
- Do **NOT** mention or refer to the AI system’s technical components or generation pipeline, such as “retrieved contexts,” “GNN,” “RAG,” “structured medical knowledge,” “knowledge graph,” or any internal data processing.
- If the evidence is insufficient or ambiguous, state this transparently, for example:
  - “The available information does not conclusively support a single diagnosis.”
  - “Symptoms may suggest several possible causes, requiring further evaluation.”
- Avoid over-explaining the origin of the knowledge; present the reasoning as if you were summarizing established medical evidence and clinical experience.

---

#### 3. SUGGESTED PLAN OF ACTION
Provide an appropriate next step, based on the confidence level, potential severity, and clinical context.
- Try answering the question 'what should i do next'
Include some or all of the following when relevant:
- Suggested **tests** (e.g., “consider complete blood count (CBC)”, “neurological exam recommended”)
- Whether to **monitor symptoms**, or seek **immediate medical attention**
- Possible **treatment options** — include only evidence-based recommendations (no speculation). This may cover:
  Clinician-directed treatments (e.g., prescription medications, procedures, therapies)
  Safe, evidence-supported self-care measures and at-home care tips the user can begin at home right now. Provide a “what to do right now” guidance- immediately actionable things the user can do while waiting for the appointment e.g., hydration, rest, dietary adjustments, over-the-counter remedies when appropriate.
- When applicable, encourage **consulting a specialist** (e.g., neurologist, pulmonologist) and expand on his next steps. Give a short “Here’s what will likely happen when you see X” (exams, possible tests).
- If symptoms are potentially **serious or progressive**, encourage seeking **prompt medical evaluation** (e.g., physician consultation)

⚠️ **If any symptoms suggest an emergency and might indicate a serious or life-threatening condition (e.g., chest pain, shortness of breath, confusion, fainting), recommend visiting the emergency department immediately.**


---

#### 4. SAFETY DISCLAIMER (ALWAYS include)
Copy and paste the following disclaimer exactly as is:

⚠️ Note, this is an AI-generated, evidence-guided clinical suggestion. It is **not a professional medical diagnosis** and should **never replace consultation with a licensed healthcare provider and medical care**. My goal is to support you with relevant information and guidance, but only a medical professional can provide a full evaluation.

---

### CRITICAL WARNINGS & STYLE GUIDE:

You MUST:
- Ground every claim in **provided data**
- Acknowledge **ambiguity or uncertainty** openly
- Speak in full, long, fluent and coherent sentences. Try to sound natural and not 'AI', with medical professionalism and empathy
- Show clear reasoning based on inputs
- Make the answer detailed and clear while remaining **concise and straightforward**, focusing on what the user cares about most: the diagnosis and the reasoning that supports it
- Use **qualifiers** like “may suggest,” “could indicate,” “is consistent with”
- If the **input symptoms are limited, ambiguous, or unclear**, note this explicitly and avoid overconfident conclusions.
- Recommend professional care when there is **any risk of under-triage**


You MUST NOT:
- Hallucinate diseases, symptoms, or treatments not found in input
- Fabricate or cite non-existent studies
- Offer casual or generic advice (e.g., “rest and fluids”) without medical justification
- Use phrases like “You should be fine” or anything falsely reassuring
- Present a definitive diagnosis
- Minimize or ignore severe or worsening symptoms
- Make up treatments, citations, or conditions not in the input
- Use medical jargon without explanation
- Mention or reference how the AI generated its response and any step of the pipeline(e.g., retrieval systems, GNN, RAG, knowledge graphs, internal processes, or pipelines)

---

### TONE:
- Cautious but confident in logical reasoning — do not speculate wildly, but explain what is likely based on the input.
- Empathetic, respectful, and aware that the user may be anxious or confused. Avoid cold or overly technical language unless it is clearly explained.
- Informative, but not overwhelming — prioritize clarity and helpfulness over medical verbosity.
- Responsible — never provide false reassurance or definitive answers when the situation is uncertain or potentially serious.
- Balanced and calm — do not use language that is alarming or anxiety-provoking (e.g., “this could be deadly”) unless medically necessary. Instead, say things like “this may indicate a condition that requires urgent evaluation.”
- Supportive, not dismissive — even if the symptoms appear mild, avoid brushing them off. Acknowledge them and offer realistic, medically informed next steps.
- Reassuring when appropriate — if symptoms are truly minor and all inputs point to low-risk explanations, it is okay to gently reassure the user — but always suggest professional confirmation.


### TARGET AUDIENCE
Assume your output will be reviewed by:
- A medical student or doctor (for reasoning clarity and accuracy)
- A technically literate user (patient or researcher)
- Your goal is to be transparent, logical, and medically responsible

---

### OUTPUT FORMAT:

Please respond as one continuous, empathetic, and professional message, beginning with a polite greeting and acknowledgment of the symptoms shared. Your response should smoothly cover the following content without explicit section titles or bullet points:
- Start by warmly thanking or acknowledging the user for sharing their symptoms, greet them and set a supportive tone. For example, “Hello, thank you for sharing these symptoms with me..” or “Hi,  I’m here to help you make sense of what you’re experiencing, based on your description.. “ or some other cordial phrasing
- Gently present the most likely diagnosis or diagnoses based on the symptom description, relevant medical evidence, and structured clinical knowledge. Use careful language such as “Based on what you’ve described, this may suggest…” or “One possible explanation is…”
- Integrate your clinical reasoning naturally in the narrative. Explain which symptoms contribute to which diagnosis, how medical sources or clinical guidelines support these conclusions, and any uncertainties involved. Use phrases like “Medical literature indicates…”, “Clinical evidence supports…”, or “These symptoms are consistent with…” Avoid mentioning the AI system’s retrieval methods or graph knowledge explicitly.
- Suggest responsible and evidence-based next steps within the same narrative. This could include recommended tests, symptom monitoring, consulting specialists, or seeking urgent care if warranted. Offer reassurance where appropriate but always emphasize the importance of professional evaluation.
- End with the safety disclaimer verbatim, gently reminding the user that this does not replace a professional medical diagnosis


### OUTPUT FORMAT EXAMPLE:

Hello, thank you for sharing these details with me. Based on the symptoms you've described, including persistent fatigue, mild shortness of breath, and occasional palpitations, one possible explanation is iron deficiency anemia. Other considerations might include thyroid dysfunction or early-stage heart-related conditions, though the current signs and symptom patterns are more consistent with a blood-related cause.

Persistent fatigue and palpitations are commonly seen in individuals with reduced red blood cell counts or low hemoglobin levels. Shortness of breath with exertion further supports a hematologic cause rather than a primarily respiratory one. Thyroid disorders can produce similar symptoms and remain a consideration. There are no clear indications of infection, significant cardiac disease, or acute respiratory illness at this stage.

It would be advisable to schedule a primary care appointment for a complete blood workup, including a complete blood count (CBC) and iron studies. If confirmed, iron supplementation may be helpful. Meanwhile, monitoring energy levels, heart rate, and breathing during mild activity could provide useful information. If symptoms worsen — especially if you notice increased heart rate, chest discomfort, or dizziness — more urgent evaluation would be necessary.

⚠️ Note, this is an AI-generated, evidence-guided clinical suggestion. It is **not a professional medical diagnosis** and should **never replace consultation with a licensed healthcare provider and medical care**. My goal is to support you with relevant information and guidance, but only a medical professional can provide a full evaluation.

"""
    return prompt

def safe_generate(model, prompt, generation_config=None, api_manager=None):
    """
    Generates content from a model safely with retry and optional API key rotation.
    """
    if api_manager:
        return api_manager.try_with_retry(model.generate_content, prompt, generation_config=generation_config).text
    else:
        while True:
            try:
                return model.generate_content(prompt, generation_config=generation_config).text
            except Exception:
                time.sleep(2)

def rag_gnn_evaluation_prompt(system_output, reference_diagnoses, symptom_description):
    prompt = f"""### Mission:
You are tasked with evaluating the quality of a diagnostic response generated by a clinical AI assistant based on a patient's symptom description.
The assistant uses a hybrid RAG-GNN architecture that combines structured medical graphs with retrieval-augmented generation from biomedical literature.

You will assess how well the output satisfies clinical reasoning, accuracy, and clarity. The evaluation should consider:
- The original **symptom description** (must be respected and directly addressed)
- The provided **reference diagnosis(es)** from physicians (very useful and assumed true but not absolute; if the system’s output is more accurate or medically reasonable, do not penalize it)
- The **retrieved biomedical evidence** included in the system's output (must be used accurately and faithfully)
- Your own **medical knowledge and general literature understanding** (to verify or refute claims)

---

### Input Case

- **Symptom Description:**
{symptom_description}

- **Reference Diagnosis(es):**
{reference_diagnoses}

- **RAG-GNN generated diagnostic response:**
{system_output}

---

### Evaluation Criteria (Total: 100 Points)

For each of the following categories, write a short explanation (1–2 sentences) about the system’s performance, then assign a score.
If a section fails to meet expectations, assign a low score — this is expected and important for fairness.

1. **Clinical Accuracy (20 pts)**
   Is the diagnosis medically plausible and aligned with the symptoms? Does it match any valid reference diagnosis or a clinically acceptable alternative?
   **Increases Points:** Uses correct medical knowledge, symptoms match diagnosis, avoids overreach, offers accurate differential diagnoses if needed.
   **Reduces Points:** Suggests implausible or unsafe diagnoses, ignores major symptoms, or contradicts established clinical understanding.

2. **Alignment with References (15 pts)**
   Does the diagnosis agree with the provided reference(s)? If not, is the deviation medically justified?
   **Increases Points:** Matches reference diagnoses or provides a clear, evidence-based reason for deviation.
   **Reduces Points:** Contradicts references without explanation, ignores provided guidance, or introduces unrelated conditions.

3. **Groundedness in Input (15 pts)**
   Is the answer directly based on the symptom description? Does it avoid irrelevant or fabricated content?
   **Increases Points:** Uses only provided symptoms, stays on-topic, avoids extra assumptions.
   **Reduces Points:** Adds symptoms not mentioned, drifts off-topic, introduces fabricated details.

4. **Use of Retrieved Evidence (10 pts)**
   Does the answer integrate relevant, reliable biomedical or clinical evidence to support the diagnosis, applying it accurately to the symptoms?
   Must not mention or reference how the answer was generated, including internal processes or pipeline components (RAG, GNN, retrieval systems, structured knowledge, etc.).
   **Increases Points:** References high-quality clinical studies, guidelines, or medical consensus; integrates evidence meaningfully into reasoning. includes at least one relevant quotation or statistic tied to the patient’s symptoms.
   **Reduces Points:** Fails to support claims with reliable evidence, hallucinates studies, misrepresents evidence, cherry-picks data, or mentions system internals.

5. **Transparency, Explainability & Informativeness (10 pts)**
   Is the reasoning clear, logical, and easy to follow? Does the response explain how the diagnosis was reached and provide useful context? Does the answer go beyond a label to offer helpful context, e.g., differential diagnoses, risk factors, caveats, or plan of action?
   **Increases Points:** Clearly links symptoms to diagnosis, explains reasoning in plain terms, offers helpful context such as differential diagnoses, risk factors, or actionable next steps. clarifies uncertainties, offers relevant at-home measures when appropriate, and **prioritizes actionable information over exhaustive explanation**.
   **Reduces Points:** Provides conclusions without explanation, or uses opaque medical jargon without clarification. shows illogical jumps, or overwhelms with irrelevant or repetitive detail that obscures the main action items.

6. **Reasoning Quality (10 pts)**
   Is the diagnostic reasoning logical and consistent? Does it show step-by-step clinical thinking supported by the input and retrieved information?
   **Increases Points:** Shows coherent reasoning, avoids contradictions, supports claims with evidence.
   **Reduces Points:** Contains logical gaps, contradicts itself, or makes unjustified leaps. Overly broad or unfocused differential, repetition.

7. **Clarity & Conciseness (10 pts)**
   Is the answer well written, easy to follow, and not unnecessarily verbose? Is it phrased politely and professionally? Does it sound human-like and avoid lists, sections, or numbering?
   **Increases Points:** Uses plain language for complex terms, is brief but complete and continuous, maintains a respectful tone, and focuses on the most relevant information (diagnosis and actionable instructions).
   **Reduces Points:** Overly verbose, repetitive, uses lists, numbering, or section headers instead of continuous narrative, includes excessive background or irrelevant details, unclear phrasing, unprofessional tone, confusing structure, or fails to focus on the critical diagnosis and next steps.
   **Note:** Long, unfocused answers (>3 short paragraphs) **should be heavily penalized** and get a reduced score. Conciseness is essential; the response should be short and to the point, and deliver a clear, direct, and actionable clinical message without unnecessary elaboration or low-priority information.

8. **Disclaimer (10 pts)**
   Does the response include a clear, explicit, medical disclaimer? This disclaimer can appear anywhere in the response and may use any reasonable wording, but it must clearly state that:
   - The response is AI-generated or not from a human clinician.
   - It is **not** a professional medical diagnosis.
   - It should **not** replace consultation with a licensed healthcare provider.
   - Only a qualified medical professional  is the only one who can provide a complete evaluation.
   **Increases Points:** Provides a complete, clear disclaimer with polite phrasing.
   **Reduces Points:** Missing any disclaimer part, incomplete, vague, or unclear wording.
---

### Evaluation Instructions

- Use **all three sources of truth**: patient symptom description, reference diagnoses, and your own verified clinical knowledge. Each should inform your assessment.
- **Carefully evaluate any cited evidence or references**: If the output mentions studies, papers, or medical facts, ensure that they are **accurately interpreted, relevant to the symptoms, and not misleading**. Hallucinations, misrepresentations, or selective quoting should be penalized.
- **Low scores are a reflection of honest evaluation, not failure**. Assign them freely when outputs do not meet criteria. Do not hesitate to give them when needed!!!
- Be fair: outputs that differ from reference diagnoses can still receive high scores if the reasoning is sound and supported by evidence.
- **Common pitfalls to watch for and penalize:**
   - Rewarding style or fluency over **medical correctness**, clarity & Conciseness cannot outweigh medical correctness. A fluent but inaccurate answer should score very low overall.
   - Including **vague, unsupported, or fabricated claims**.
   - Using evidence incorrectly, cherry-picking results, or overstating findings.
   - Overconfidence when the evidence is ambiguous or weak.
   - Producing overly long, unfocused answers that dilute urgent or critical information.
   - Mentioning internal systems, pipelines, or technical processes (e.g., RAG, GNN, retrieval methods).
   - Writing in sections or with headers/topics (answer should be continous and fluent).

- **Encourage concise, focused reasoning**, especially for potentially urgent medical scenarios.
  - Answers should prioritize critical symptoms, likely diagnoses, and actionable next steps rather than exhaustive lists of possibilities. penlize long lists or repetition.

- **Assess reasoning and transparency.**
  - The output should explain **why each diagnosis is considered**, linking symptoms, risk factors, or patterns to conclusions.
  - Penalize outputs that list conditions without rationale or that show illogical or inconsistent reasoning.
  - Step-by-step reasoning, showing how evidence supports the hypothesis, is highly valued.

- **Evaluate informativeness and guidance.**
  - Look for actionable next steps, tests, monitoring advice, red-flag identification, or referrals to specialists.
  - Penalize excessive low-priority details or long, unfocused lists that dilute critical guidance.

- **Consider focus, conciseness, and urgency.**
  - Responses should be **to the point**, highlighting the most likely diagnoses, critical symptoms, and recommended actions.
  - Penalize repeated points, unnecessary background, or overly long narratives that may overwhelm users in urgent scenarios.

- **Check for avoidance of internal system references.**
  - Outputs must **not mention internal pipelines or tools** (e.g., RAG, GNN, knowledge graphs).
  - Penalize any mention of how the answer was generated.

- **Be fair and transparent in scoring.**
  - Low scores are **not failures**; they are essential for honest evaluation.
  - Clearly document reasons for point deductions.
  - Ensure scoring reflects both accuracy and clarity, but never reward style over substance.

- **patient safety:** Unsafe, harmful, or dangerous recommendations automatically result in ≤10 points, even if other aspects are strong


---

- For each category, use the format:

[index].[Category Name]:(X/Max_points): [Explenation]
- Keep the explenation straight-foward and concise
- Ensure the sum of the maximum **possible** points equals exactly 100.

- Then show the sum calculation over all category scores under 'Total Score Calculation: '
- for example if we got: Clinical Accuracy: 18/20, Alignment with References: 12/15, Groundedness in Input: 14/15, Use of Retrieved Evidence: 8/10, Transparency, Explainability & Informativeness: 7/10, Reasoning Quality: 8/10, Clarity & Conciseness: 9/10, Disclaimer: 0/10
  Total Score Calculation: 18+12+14+8+7+8+9+0 = 76


- ALWAYS Finish with the final score (sum over all categories) in this exact format: 'Final Score: X/100'


### IMPORTANT NOTE !!!!!:
- The very last line of your response MUST be exactly: 'Final Score: X/100'
- Replace X with the total score which MUST be the sum over all category scores. Do NOT include decimals.
- Make sure that its ACTUALLY the combined score and not just some random number!
- Do NOT add any text, punctuation, or explanation after this line. DO NOT OMMIT OR SKIP THIS LINE!
- **If you do not follow this exact finale score format, the evaluation will be considered invalid.**
---

Now produce your evaluation - and again **make sure** to end with 'Final Score: X/100'.
"""
    return prompt

# def extract_final_diagnosis_score(text):
#     match = re.search(r"Final Score:\s*(\d{1,3})\s*/\s*100", text)
#     if match:
#         return int(match.group(1))
#     else:
#         return None


def extract_final_diagnosis_score(text):
    match = re.search(r"Final Score[:\s]*\(?(\d{1,3})(?:\s/\s*100)?\)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_clinical_accuracy_score(text):
    match = re.search(r"Clinical Accuracy:?\s*\(?(\d{1,2})\s*/\s*20\)?", text)
    return int(match.group(1)) if match else None

def extract_use_of_retrieved_evidence_score(text):
    match = re.search(r"Use of Retrieved Evidence:?\s*\(?(\d{1,2})\s*/\s*10\)?", text)
    return int(match.group(1)) if match else None

def extract_transparency_explainability_informativeness_score(text):
    match = re.search(
        r"Transparency, Explainability\s*&\s*Informativeness:?\s*\(?(\d{1,2})\s*/\s*10\)?", 
        text
    )
    return int(match.group(1)) if match else None

from multiprocessing import Lock
api_key_lock = Lock()

def run_generation(query, gnn_terms, retrieved_contexts, retrieved_contexts_no_gnn=None, reference_diagnosis=None, testing=True, temperature=0.2):
    """
    Unified RAG-GNN pipeline.
    - testing=False: just generate RAG-GNN content
    - testing=True: generate + baseline + evaluation + score comparison
    """
    api_key_lock.acquire()
    try:
        api_manager.configure_next()
        cur_api_manager = api_manager.__copy__()
    finally:
        api_key_lock.release()

    model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config={"temperature": temperature}
    )

    # Step 2: RAG-GNN generation
    gen_prompt = rag_gnn_generation_prompt(query, retrieved_contexts, gnn_terms)
    rag_gnn_output = safe_generate(model, gen_prompt, api_manager=api_manager)

    # If just generating content
    if not testing:
        return {
            "query": query,
            "gnn_terms": gnn_terms,
            "retrieved_contexts": retrieved_contexts,
            "rag_gnn_output": rag_gnn_output
        }

    # Testing mode requires reference diagnosis
    if reference_diagnosis is None:
        raise ValueError("reference_diagnosis is required in testing mode.")

    # Baseline generation
    baseline_output = safe_generate(model, query, api_manager=cur_api_manager)
    baseline_rag_output = safe_generate(model, rag_gnn_generation_prompt(query, retrieved_contexts_no_gnn, ""), api_manager=cur_api_manager)

    # Evaluation prompts
    rag_eval_prompt = rag_gnn_evaluation_prompt(rag_gnn_output, reference_diagnosis, query)
    base_eval_prompt = rag_gnn_evaluation_prompt(baseline_output, reference_diagnosis, query)
    base_eval_rag_prompt = rag_gnn_evaluation_prompt(baseline_rag_output, reference_diagnosis, query)

    rag_eval = safe_generate(model, rag_eval_prompt, api_manager=cur_api_manager)
    base_eval = safe_generate(model, base_eval_prompt, api_manager=cur_api_manager)
    base_eval_rag = safe_generate(model, base_eval_rag_prompt, api_manager=cur_api_manager)

    rag_score = extract_final_diagnosis_score(rag_eval)
    base_score = extract_final_diagnosis_score(base_eval)
    base_rag_score = extract_final_diagnosis_score(base_eval_rag)

    rag_clinical_accuracy = extract_clinical_accuracy_score(rag_eval)
    base_clinical_accuracy = extract_clinical_accuracy_score(base_eval)
    base_rag_clinical_accuracy = extract_clinical_accuracy_score(base_eval_rag)

    rag_use_of_retrieved_evidence = extract_use_of_retrieved_evidence_score(rag_eval)
    base_use_of_retrieved_evidence = extract_use_of_retrieved_evidence_score(base_eval)
    base_rag_use_of_retrieved_evidence = extract_use_of_retrieved_evidence_score(base_eval_rag)

    rag_transparency = extract_transparency_explainability_informativeness_score(rag_eval)
    base_transparency = extract_transparency_explainability_informativeness_score(base_eval)
    base_rag_transparency = extract_transparency_explainability_informativeness_score(base_eval_rag)

    diff = (rag_score or 0) - (base_score or 0)

    # RAG_GNN_scores.append(rag_score)
    # Vanilla_LLM_scores.append(base_score)

    stats = {
        "rag_gnn_score": rag_score,
        "baseline_score": base_score,
        # "difference": diff,
        "rag_score": base_rag_score,
    }

    return {
        "query": query,
        "reference_diagnosis": reference_diagnosis,
        "gnn_terms": gnn_terms,
        "retrieved_contexts": retrieved_contexts,
        "rag_gnn_output": rag_gnn_output,
        "baseline_output": baseline_output,
        "rag_eval": rag_eval,
        "baseline_eval": base_eval,
        "rag_gnn_results" : {
            "score": rag_score,
            "clinical_accuracy": rag_clinical_accuracy,
            "use_of_retrieved_evidence": rag_use_of_retrieved_evidence,
            "transparency": rag_transparency
        },
        "baseline_results": {
            "score": base_score,
            "clinical_accuracy": base_clinical_accuracy,
            "use_of_retrieved_evidence": base_use_of_retrieved_evidence,
            "transparency": base_transparency
        },
        "baseline_rag_results": {
            "score": base_rag_score,
            "clinical_accuracy": base_rag_clinical_accuracy,
            "use_of_retrieved_evidence": base_rag_use_of_retrieved_evidence,
            "transparency": base_rag_transparency
        }
    }