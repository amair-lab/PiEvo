import json
import os
from typing import Dict, Any, List, Tuple, Optional, Union


def _compact_candidate_json(obj) -> str:
    """Serialize a candidate dict to JSON with 2D number arrays kept on single lines.

    This avoids vertically exploding large pixel_mask arrays (e.g. 16x16 = 256 lines
    with indent=2) while keeping the full data scientifically complete.
    """
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append(f"  {json.dumps(k)}: {_compact_candidate_json(v)}")
        return "{\n" + ",\n".join(items) + "\n}"
    elif isinstance(obj, list) and obj and all(isinstance(x, int) for x in obj):
        return json.dumps(obj)
    elif (
        isinstance(obj, list)
        and obj
        and all(
            isinstance(x, list) and all(isinstance(y, (int, float)) for y in x)
            for x in obj
        )
    ):
        rows = [json.dumps(row) for row in obj]
        return "[\n    " + ",\n    ".join(rows) + "\n  ]"
    elif isinstance(obj, list):
        return json.dumps(obj, indent=2)
    else:
        return json.dumps(obj)


def get_principle_guidance_prompt(
    anomalies: List[Tuple[str, float, float, float]],  # compute by the prediction of y
    num_principles: int,
    principle_beliefs: Dict[str, float],
    top_principle_text: str = "",
    principles: Dict[
        str, str
    ] = None,  # all principles with  key and its content (value)
) -> Tuple[str, str]:
    max_belief = 0.0
    if principle_beliefs:
        max_belief = max(principle_beliefs.values())

    anomaly_text = ";\n".join(
        [
            f"  - For hypothesis '{h}', expected outcome {p:.2f}, but observed {o:.2f} (surprisal: {s:.3f})"
            for h, o, p, s in anomalies[: int(os.environ["ANOMALY_COUNTS_THRESHOLD"])]
        ]
    )

    formatted_existing_principles = "\n".join(
        [f'- [{k}]:  "{principles[k]}"; ' for k, v in principle_beliefs.items()]
    )

    if num_principles < int(os.environ["NUM_EXPLORATION_CASES"]):
        guidance_type = "DIVERSIFY_PRINCIPLE"
        guidance_message = f"""
# STATUS: DIVERSIFYING PRINCIPLE SPACE
We need more diverse ideas to explore the problem space.

# CONTEXT: EXISTING PRINCIPLES AND BELIEFS
{formatted_existing_principles if formatted_existing_principles else "No principles exist yet."}

# YOUR OBJECTIVE: DIVERSIFY
    - Propose 1 new, creative principle that can solve the problem (i.e., yield high outcome).
    - Think of mechanisms that are entirely **different/distant** from the ones we already have.
    - One principle must contain the whole logic of its premises and rational reasoning context. 
    - Incremental or minor modifications of previous principles are strongly forbidden. 
"""

    elif len(anomalies) >= float(os.environ["ANOMALY_COUNTS_THRESHOLD"]):
        if max_belief >= float(os.environ["HIGH_CONFIDENCE_THRESHOLD"]):
            guidance_type = "REFINE_FROM_ANOMALY"
            guidance_message = f"""
# STATUS: HIGH CONFIDENCE, BUT MANY ANOMALIES DETECTED
CURRENT TOP-POTENTIAL PRINCIPLE: "{top_principle_text}"

This principle is performing well, but it fails to explain the following persistent anomalies:
{anomaly_text}

# CONTEXT: CURRENT WORKING SET OF PRINCIPLES
{formatted_existing_principles}

# YOUR OBJECTIVE: REFINE PRINCIPLES, DO NOT REPLACE
    - **Do NOT propose a totally new principle.** Just to refine based on the current evidence. 
    - Your task is to propose a **refinement** or **variant** of the top principle (by considering the other principles).
    - Focus on what these anomalies have in common, the union pattern and the specificities.
    - Propose an updating (e.g., add a condition/exception, extend a rule or replace some conditions) that explains these failures *without* invalidating the principle's core success.
"""

        else:
            guidance_type = "PROPOSE_BY_ANOMALY"
            guidance_message = f"""
# STATUS: LOW CONFIDENCE AND SYSTEMATIC FAILURES - WE ARE IN THE WRONG TRACK
CURRENT PRINCIPLES ARE FAILING.

We have detected significant anomalies that our current principles cannot explain:
{anomaly_text}

# CONTEXT: THESE ARE THE PRINCIPLES THAT FAILED/INCOMPLETE/BIASED
{formatted_existing_principles}

# YOUR OBJECTIVE IS TO PROPOSE A NEW PRINCIPLE BY FOLLOWING THE GUIDANCE:
    - **This is a discovery task.** Our existing ideas are insufficient.
    - Analyze the *pattern* in these failures, and connect to knowledge that support these observations.
    - Propose a **new, creative principle** that can explain why these outcomes occurred.
    - The new principle should ideally account for patterns missed by all existing principles.
    - To achieve better interpretability, the new proposed principle MUST align with the observations (candidates paired with observed outcomes). 
"""

    else:
        guidance_type = "PAUSE_PRINCIPLES"

        if max_belief >= float(os.environ["HIGH_CONFIDENCE_THRESHOLD"]):
            pause_reason = (
                "CURRENTLY EXPLOITATION mode, focusing on the high-potential principle."
            )
        else:
            pause_reason = "CURRENTLY EXPLORATION mode, gathering more data to differentiate existing principles."

        guidance_message = f"""
# STATUS: PAUSING PRINCIPLE GENERATION
REASON: {pause_reason}

The system does not require new principles at this moment. 

# YOUR ACTION: SKIP THIS TURN
**SAY `PRINCIPLE_SUBMISSION` ONLY.**
(Do not add any other words or sentences.)
""".replace("        ", "")

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
    _task = task.split("## Last record of the experiment")[0].strip()

    max_belief = 0.0

    if principle_beliefs:
        sorted_beliefs = sorted(
            principle_beliefs.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_beliefs:
            max_belief = sorted_beliefs[0][1]

    if candidates_done:
        candidates_done = sorted(
            candidates_done, key=lambda c: c["outcome"], reverse=True
        )

    if candidates_done:
        candidates_done = sorted(
            candidates_done, key=lambda c: c["outcome"], reverse=True
        )

        candidate_entry_lines = []
        for c in candidates_done:
            candidate = c.get("candidate", c)
            outcome = c.get("outcome", 0.0)
            reasoning = c.get("reasoning", "")
            candidate_json = _compact_candidate_json(candidate)

            entry = f"### Candidate (outcome: {outcome:.6f}):\n```json\n{candidate_json}\n```"
            if reasoning:
                entry += f"\n**Experimenter's Analysis/Reasoning:**\n{reasoning}"
            candidate_entry_lines.append(entry)

        candidates_summary = "\n\n".join(candidate_entry_lines)

        diversity_instruction = """
## DIVERSITY REQUIREMENT [CRITICAL]:
Your new hypotheses MUST explore regions of the design space NOT covered above.
Specifically:
- Do NOT reuse the same rationale. Each hypothesis must differ in its core mechanism, not just minor parameter tweaks.
- Propose hypotheses that are ORTHOGONAL to the tested ones — change multiple dimensions simultaneously.
- Use the **Experimenter's Analysis** (if available) to avoid failure modes and target promising visual features.
- Repeating similar hypotheses will waste the experiment budget and hinder scientific discovery.
"""
    else:
        candidates_summary = "No candidates tested yet."
        diversity_instruction = """
## DIVERSITY REQUIREMENT [CRITICAL]:
Since no candidates have been tested yet, maximize coverage of the design space.
Each hypothesis should explore a distinctly different mechanism or configuration.
Do NOT cluster all hypotheses around a single pattern.
"""

    # ------------------------ Exploitation -----------------------
    if max_belief >= float(
        os.environ["HIGH_CONFIDENCE_THRESHOLD"]
    ) and num_principles >= int(os.environ["NUM_EXPLORATION_CASES"]):
        return (
            "EXPLOITATION",
            f"""
[SYSTEM TASK]

{_task}

# HIGH CONFIDENCE (EXPLOITING)

## MEMORY (WHAT HAVE BEEN TESTED):
{candidates_summary}

{diversity_instruction}

## FOCUS PRINCIPLE:
"{top_principle_text}"

## INSTRUCTION:
Following this principle, propose **THREE** novel candidate hypotheses in JSON format.
We are confident in this principle and have explored alternatives.
Our goal is to **maximize positive experiment outcomes** based on it. Therefore, your task now is to make full use of this scientific principle, to propose candidate with higher and higher outcome.

## REQUIREMENTS:
    - Propose 3 **different** and untested candidates (A, B, C) that you predict will yield **higher reward**.
    - DO NOT propose any hypothesis from the memory list (repeating candidates will be strongly rejected).
    - Each candidate must differ from the others in at least TWO key parameters — do not just tweak one value.
    - Each candidate must differ from ALL tested candidates in the memory above. If a candidate looks similar to any tested one, redesign it.
    - The existing EXPLOITATION PHASE MEMORY can help you to propose novel hypotheses.
    - Consider the patterns of candidates and their corresponding outcome values.
    - Remember to assign external `HYPOTHESIS_SUBMISSION` after your JSON submission is complete.
""".replace("        ", ""),
        )

    # ------------------------ Exploration ------------------------
    else:
        if second_best_principle_text:
            task_title = "LOW CONFIDENCE (COMPETING EXPLORATION)"
            task_description = f"""
[Important] We are in a 'Scientific Debate' phase.
Our goal is NOT just to get a high score, but to **differentiate** between conflicting principles.
We need experiments where our top principles make **drastically different predictions** while seeking high outcome.

# PRINCIPLE 1 (Top Belief):
"{top_principle_text}"

# PRINCIPLE 2 (Runner-up Belief):
"{second_best_principle_text}"

# REQUIREMENTS:
Your JSON submission must contain exactly 4 hypotheses (A, B, C, D).

1.  **HYPOTHESIS_SUBMISSION_A and B (The "Controversial" Candidates):**
    - RATIONAL: Propose hypotheses where **Principle 1 predicts a HIGH outcome**, but **Principle 2 might predict a LOW outcome** (or vice versa).
    - Focus on the *distinctive features* of the principles (e.g., if P1 relies on geometry and P2 relies on resonance, find a geometry that kills resonance but keeps asymmetry).

2.  **HYPOTHESIS_SUBMISSION_C and D (The "Boundary" Testers):**
    - RATIONAL: Explore the **edges of the parameter space** or configurations we haven't touched.
    - Do NOT play it safe. We need to know if the principles hold up in extreme conditions.
    - Even if the expected outcome is uncertain, the Information Gain will be high.

Note:
- Do NOT simply cluster around previous successful patterns.
- Each hypothesis must differ from the others in at least TWO key parameters.
- Try to vary the structural configuration, symmetry, and key parameters significantly.
"""

        else:
            task_title = "LOW CONFIDENCE (SINGLE-PRINCIPLE STRESS TEST)"
            task_description = f"""
Our goal is to **Stress Test** our ONLY principle.
We don't just want to confirm it works (we know that); we want to find **where it breaks**.
Finding the failure mode of a principle is just as valuable as finding a success.

# DIRECTED PRINCIPLE:
"{top_principle_text}"

# REQUIREMENTS:
Your JSON submission must contain 3-4 different hypotheses (A, B, C).

1.  **"Extreme" Hypothesis (A):**
    - RATIONAL: Push the parameters to **extreme configurations**. See if the principle's logic still holds.

2.  **"Counter-Intuitive" Hypothesis (B):**
    - RATIONAL: Propose a configuration that looks "weird" or "unlikely" according to standard intuition, but is theoretically interesting under the directed principle.

3.  **"Gap Filling" Hypothesis (C):**
    - RATIONAL: Look at the tested candidates below. Find a large "hole" in the design space (e.g., a parameter value or configuration we haven't touched) and place a sample there.
"""

        return (
            "EXPLORATION",
            f"""
# Task

{_task}

# {task_title}

## Candidates have been tested (avoid to propose again):
{candidates_summary}

{diversity_instruction}

## INSTRUCTION:
Provide candidate hypotheses in JSON format.
{task_description}

## REQUIREMENTS:
- [STRICT] Use the structure of analytic reasoning with principles.
- [CRITICAL] **DIVERSITY IS KEY.** Each hypothesis must differ from ALL tested candidates AND from each other in at least TWO key parameters.
- [CRITICAL] Analyze the "Candidates have been tested" section above. Identify which parameter values and structural patterns have been exhausted, then propose hypotheses that fill the gaps.
- [CRITICAL] **NO REPETITIONS.** Do NOT propose any candidate structurally similar to those already tested. The system wastes rounds when you repeat configurations — this blocks all scientific progress.
- [CRITICAL] **EXPLORE WIDELY.** If you find yourself proposing something close to a tested candidate, stop and redesign it to be structurally different. Change the pixel_mask pattern entirely, not just a few pixels.
- [FORM] All hypotheses must be testable.
- [FORM] Remember to assign an external `HYPOTHESIS_SUBMISSION` signal.
""",
        )


def get_experiment_guidance_prompt(selected_hypothesis, task: str) -> str:
    """
    Returns the experiment guidance prompt for testing a hypothesis.

    Args:
        selected_hypothesis: Either a dict (structured candidate from Hypothesis Agent)
            or a string (legacy key=value format).
        task: The full task description.

    Returns:
        guidance_message
    """
    if isinstance(selected_hypothesis, dict):
        candidate_display = _compact_candidate_json(selected_hypothesis)
    else:
        candidate_display = str(selected_hypothesis)

    return f"""
[INSTRUCTION]
You MUST now execute the experiment tool on the following selected candidate.
- You MUST invoke the tool by passing each key-value pair as a separate keyword argument.
- Do NOT stringify the candidate. Do NOT pass the entire dict as a single argument.

[Selected candidate to test]
```
{candidate_display}
```

[Execution and Analysis]
1. Execute the tool with the candidate parameters.
2. **Analyze the Visual Feedback**: The tool will return a visualization (e.g., band structure, Berry curvature). You MUST examine this image carefully.
3. **Reasoning**: Discuss what you see in the visualization. Does it show a gap opening? Is there a topological transition? How does the visual evidence compare to the numerical score?
4. **Submission**: After your analysis, submit the final result using the `EXPERIMENT_SUBMISSION` token with the predefined JSON format.

[Constraints you must follow]
- After the tool returns, include the tool's numerical output as the \"outcome\" in your EXPERIMENT_SUBMISSION JSON.
- The \"candidate\" field in your JSON must be the exact dict/object you were given to test.
- Do NOT write principle text or descriptions inside the JSON "candidate" field.
""".replace("    ", "")
