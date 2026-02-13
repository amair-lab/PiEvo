import os
from typing import Dict, Any, List, Tuple, Optional



def get_principle_guidance_prompt(
        anomalies: List[Tuple[str, float, float, float]],  # compute by the prediction of y
        num_principles: int,
        principle_beliefs: Dict[str, float],
        top_principle_text: str = "",
        principles: Dict[str, str] = None,  # all principles with  key and its content (value)
) -> Tuple[str, str]:
    max_belief = 0.0
    if principle_beliefs:
        max_belief = max(principle_beliefs.values())

    anomaly_text = ";\n".join(
        [f"  - For hypothesis '{h}', expected outcome {p:.2f}, but observed {o:.2f} (surprisal: {s:.3f})"
         for h, o, p, s in anomalies[:int(os.environ["ANOMALY_COUNTS_THRESHOLD"])]]
    )

    formatted_existing_principles = "\n".join(
        [f"- [{k}]:  \"{principles[k]}\"; "
         for k, v in principle_beliefs.items()]
    )

    if num_principles < int(os.environ["NUM_EXPLORATION_CASES"]):
        guidance_type = "DIVERSIFY_PRINCIPLE"
        guidance_message = f"""
STATUS: DIVERSIFYING PRINCIPLE SPACE
We need more diverse ideas to explore the problem space.

CONTEXT: EXISTING PRINCIPLES AND BELIEFS
{formatted_existing_principles if formatted_existing_principles else "No principles exist yet."}

YOUR OBJECTIVE: DIVERSIFY
    - Propose 1 new, creative principle that can solve the problem (i.e., yield high outcome).
    - Think of mechanisms that are entirely different/distant from the ones we already have.
    - One principle must contain the whole logic of its premises and rational reasoning context. 
    - Incremental or minor modifications of previous principles are strongly forbidden. 
"""

    elif len(anomalies) >= float(os.environ["ANOMALY_COUNTS_THRESHOLD"]):
        if max_belief >= float(os.environ["HIGH_CONFIDENCE_THRESHOLD"]):
            guidance_type = "REFINE_FROM_ANOMALY"
            guidance_message = f"""
STATUS: HIGH CONFIDENCE, BUT MANY ANOMALIES DETECTED
CURRENT TOP-POTENTIAL PRINCIPLE: "{top_principle_text}"

This principle is performing well, but it fails to explain the following persistent anomalies:
{anomaly_text}

CONTEXT: CURRENT WORKING SET OF PRINCIPLES
{formatted_existing_principles}

YOUR OBJECTIVE: REFINE PRINCIPLES, DO NOT REPLACE
    - Do NOT propose a totally new principle. Just to refine based on the current evidence. 
    - Your task is to propose a refinement or variant of the top principle (by considering the other principles).
    - Focus on what these anomalies have in common, the union pattern and the specificities.
    - Propose an updating (e.g., add a condition/exception, extend a rule or replace some conditions) that explains these failures *without* invalidating the principle's core success.
"""
        
        else:
            guidance_type = "PROPOSE_BY_ANOMALY"
            guidance_message = f"""
STATUS: LOW CONFIDENCE AND SYSTEMATIC FAILURES - WE ARE IN THE WRONG TRACK
CURRENT PRINCIPLES ARE FAILING.

We have detected significant anomalies that our current principles cannot explain:
{anomaly_text}

CONTEXT: THESE ARE THE PRINCIPLES THAT FAILED/INCOMPLETE/BIASED
{formatted_existing_principles}

YOUR OBJECTIVE IS TO PROPOSE A NEW PRINCIPLE BY FOLLOWING THE GUIDANCE:
    - This is a discovery task. Our existing ideas are insufficient.
    - Analyze the *pattern* in these failures, and connect to knowledge that support these observations.
    - Propose a new, creative principle that can explain why these outcomes occurred.
    - The new principle should ideally account for patterns missed by all existing principles.
    - To achieve better interpretability, the new proposed principle MUST align with the observations (candidates paired with observed outcomes). 
"""

    else:
        guidance_type = "PAUSE_PRINCIPLES"

        if max_belief >= float(os.environ["HIGH_CONFIDENCE_THRESHOLD"]):
            pause_reason = "CURRENTLY EXPLOITATION mode, focusing on the high-potential principle."
        else:
            pause_reason = "CURRENTLY EXPLORATION mode, gathering more data to differentiate existing principles."

        guidance_message = f"""
STATUS: PAUSING PRINCIPLE GENERATION
REASON: {pause_reason}

The system does not require new principles at this moment. 

YOUR ACTION: SKIP THIS TURN
SAY `PRINCIPLE_SUBMISSION` ONLY.
(Do not add any other words or sentences.)
"""

    return guidance_type, guidance_message








def get_hypothesis_guidance_prompt(
        task: str,
        top_principle_text: str,
        second_best_principle_text: str,
        principle_beliefs: Dict[str, float],
        num_principles: int,
        candidates_done: List[Dict[str, float]] = None,
        exploitation_candidates: List[Dict[str, float]] = None,
        principles: Dict[str, str] = None,
) -> Tuple[str, str]:
    max_belief = 0.0

    if principle_beliefs:
        sorted_beliefs = sorted(principle_beliefs.items(), key=lambda item: item[1], reverse=True)
        if sorted_beliefs:
            max_belief = sorted_beliefs[0][1]

    if candidates_done:
        candidates_done = sorted(candidates_done, key=lambda c: c['outcome'], reverse=True)
        candidates_done_text = "\n".join(f"- {c['candidate']} \t Outcome: {c['outcome']}" for c in candidates_done if  c['outcome'] != 0.0)
    else:
        candidates_done_text = "No candidates tested yet."

    # ------------------------ Exploitation -----------------------
    if max_belief >= float(os.environ["HIGH_CONFIDENCE_THRESHOLD"]) and num_principles >= int(os.environ["NUM_EXPLORATION_CASES"]):
        return "EXPLOITATION", \
f"""
[SYSTEM TASK]

{task}

# HIGH CONFIDENCE (EXPLOITING)

MEMORY (WHAT HAVE BEEN TESTED - DO NOT REPEAT):
{candidates_done_text}

FOCUS PRINCIPLE: 
"{top_principle_text}"

INSTRUCTION: 
Following this principle, propose THREE novel candidate hypotheses in JSON format.
We are confident in this principle and have explored alternatives. 
Our goal is to maximize positive experiment outcomes based on it. Therefore, your task now is to make full use of this scientific principle, to propose candidate with higher and higher outcome. 

REQUIREMENTS:
    - Propose 3 different and untested candidates (A, B, C) that you predict will yield higher reward.
    - DO NOT propose any hypothesis from the memory list (repeating candidates will be strongly rejected).
    - The existing EXPLOITATION PHASE MEMORY can help you to propose novel hypotheses.
    - Consider the patterns of candidates and their corresponding outcome values. 
    - Remember to assign external `HYPOTHESIS_SUBMISSION` after your JSON submission is complete.
"""

    # ------------------------ Exploration ------------------------
    else:
        if second_best_principle_text:
            task_title = "LOW CONFIDENCE (COMPETING EXPLORATION)"
            task_description = \
f"""
[Important] We are in a 'Scientific Debate' phase. 
Our goal is NOT just to get a high score, but to differentiate between conflicting principles.
We need experiments where our top principles make drastically different predictions.

PRINCIPLE 1 (Top Belief):
"{top_principle_text}"

PRINCIPLE 2 (Runner-up Belief):
"{second_best_principle_text}"

REQUIREMENTS:
Your JSON submission must contain exactly 4 hypotheses (A, B, C, D).

1.  HYPOTHESIS_SUBMISSION_A and B (The "Controversial" Candidates):
    - RATIONAL: Propose hypotheses where Principle 1 predicts a HIGH outcome, but Principle 2 might predict a LOW outcome (or vice versa). 
    - Focus on the *distinctive features* of the principles (e.g., if P1 relies on parameter A and P2 relies on parameter B, find a combination that satisfies A but challenges B).

2.  HYPOTHESIS_SUBMISSION_C and D (The "Boundary" Testers):
    - RATIONAL: Explore the edges of the parameter space or combinations we haven't touched.
    - Do NOT play it safe. We need to know if the principles hold up in extreme conditions (e.g., boundary values of parameters).
    - Even if the expected outcome is uncertain, the Information Gain will be high.

Note: 
- Do NOT simply cluster around previous successful parameters. 
- Try to vary parameters significantly to test the robustness of the principles.
"""

        else:
            task_title = "LOW CONFIDENCE (SINGLE-PRINCIPLE STRESS TEST)"
            task_description = \
f"""
Our goal is to Stress Test our ONLY principle. 
We don't just want to confirm it works (we know that); we want to find where it breaks.
Finding the failure mode of a principle is just as valuable as finding a success.

DIRECTED PRINCIPLE:
"{top_principle_text}"

REQUIREMENTS:
Your JSON submission must contain 3-4 different hypotheses (A, B, C).

1.  "Extreme" Hypothesis (A):
    - RATIONAL: Push one or two parameters to their allowed limits (min or max). See if the principle's logic still holds at the extremes.

2.  "Counter-Intuitive" Hypothesis (B):
    - RATIONAL: Propose a configuration that looks "weird" or "unlikely" according to standard intuition, but is theoretically interesting under the directed principle.

3.  "Gap Filling" Hypothesis (C):
    - RATIONAL: Look at the tested candidates below. Find a large "hole" in the parameter space (e.g., a range of parameters we haven't touched) and place a sample there.
"""

        return "EXPLORATION", \
f"""
{task}

{task_title}

Candidates have been tested (avoid to propose again):
{candidates_done_text}

INSTRUCTION: 
Provide candidate hypotheses in JSON format.
{task_description}

REQUIREMENTS:
- [STRICT] Use the structure of analytic reasoning with principles.
- [CRITICAL] DIVERSITY IS KEY. Do not submit hypotheses that look similar to the "Candidates have been tested". Change the parameters significantly!
- [FORM] All hypotheses must be testable.
- [FORM] Remember to assign an external `HYPOTHESIS_SUBMISSION` signal.
"""





def get_experiment_guidance_prompt(
        selected_hypothesis: str,
        task: str
) -> str:
    """
    Returns the experiment guidance prompt for testing a hypothesis.
    Adapts the prompt based on the phase (IDS vs outcome optimization).

    Returns:
        guidance_message
    """

    def parse_params_string(params_str: str) -> dict:
        """
        Parse parameter string into dictionary.

        Args:
            params_str: String like "x=20.0, y=20.0, z=10.0"

        Returns:
            Dictionary with parameter names as keys and float values
        """
        result = {}

        pairs = params_str.split(',')

        for pair in pairs:
            pair = pair.strip()
            if '=' in pair:
                key, value = pair.split('=')
                key = key.strip()
                value = value.strip()
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value

        return result

    if "=" in selected_hypothesis and ", " in selected_hypothesis:
        selected_hypothesis = parse_params_string(selected_hypothesis)

    return f"""
    
TASK YOU ARE DOING (YOU ARE THE EXPERIMENT AGENT)

{task}


INSTRUCTION: 
Execute hypothesis with the given tool.

Candidate hypotheses: 

    - {selected_hypothesis} 
    
    WARNING: To correctly use the tool, you MUST parse it to fit the tool's parameter list.
    
    e.g., if hypotheses is "a=x, b=y, c=z", then the tool call should be {{"a": "x", "b": "y", "c": "z"}} for including all parameters and their values. 

Note: 
    - You can ONLY test this suggested hypothesis candidate, others are not allowed to be tested. 
    - After the experiment, please submit the results with `EXPERIMENT_SUBMISSION` signal. 
    - Again, remember to assign the EXPERIMENT_SUBMISSION token after getting the tool yielded results.

Know that there are only three possible results:
- If the tool works successfully, return the output as the outcome for this hypothesis, and submit the JSON block with external `EXPERIMENT_SUBMISSION` signal. 
- If the tool detects invalid input, return 0.0 for this hypothesis, then submit the JSON block with external `EXPERIMENT_SUBMISSION` signal.
- If no candidate hypotheses provided, please submit by `EXPERIMENT_SUBMISSION` without JSON block, and instruct the Hypothesis Agent to propose novel hypotheses. 
"""