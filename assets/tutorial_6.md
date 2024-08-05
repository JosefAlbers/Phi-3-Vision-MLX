# Part 6: Implementing Constrained Decoding for Phi-3-Vision

## Introduction

In this tutorial, we'll look at constrained decoding, a powerful technique for guiding the text generation of our Phi-3-Vision model. This method allows us to control the model's output structure and content without altering its underlying knowledge or capabilities. Constrained decoding is particularly useful for tasks requiring specific output formats or structured reasoning.

## Understanding Constrained Decoding

Constrained decoding works by setting "constraints" - specific phrases or structures that the model must include in its output within a certain number of tokens. This approach offers several benefits:

1. **Structured Output**: Ensures generated text follows a predetermined format.
2. **Guided Reasoning**: Encourages step-by-step thought processes.
3. **Consistent Responses**: Helps maintain uniformity across multiple generations.
4. **Error Reduction**: Minimizes off-topic or irrelevant content.

Common applications include:

- Generating code with specific elements
- Creating responses in particular formats (e.g., JSON, XML)
- Producing step-by-step reasoning for problem-solving
- Answering multiple-choice questions with explanations

## Implementing Constrained Decoding

Let's examine a pseudocode implementation of constrained decoding:

```python
def constrain(model, processor, prompt, constraints):
    input_ids = process(prompt)
    for each constraint in constraints:
        max_tokens, constraint_text = constraint
        constraint_ids = tokenize(constraint_text)
        
        best_sequence = input_ids
        best_score = -infinity
        
        for token_count = 1 to max_tokens:
            candidate_sequences = generate_candidates(best_sequence)
            for each candidate in candidate_sequences:
                full_sequence = concatenate(candidate, constraint_ids)
                score = calculate_sequence_score(full_sequence)
                
                if score > best_score:
                    best_score = score
                    best_sequence = candidate
            
            if best_sequence ends with constraint_ids:
                break
        
        input_ids = concatenate(best_sequence, constraint_ids)
    
    return decode(input_ids)
```

This pseudocode outlines the core logic of constrained decoding. Here's how it works:

1. We start with the initial prompt.
2. For each constraint:

    - We generate candidate sequences up to the max token limit.
    - For each candidate, we calculate the score of the candidate plus the constraint.
    - We keep track of the best-scoring sequence.
    - If the best sequence naturally ends with the constraint, we stop early.
    - Otherwise, we force-append the constraint after reaching max tokens.
   
3. We return the final generated text.

This implementation allows for flexibility in how we apply constraints. It tries to generate text that naturally includes the constraints, but if it can't do so within the token limit, it ensures the constraints are still included.

It's worth noting that this is a simplified version of the algorithm. In practice, you might need to adjust this based on your specific model architecture and requirements. For example, you might want to implement beam search or adjust how scores are calculated for better results.

## Using Constrained Decoding

Let's look at two practical applications of constrained decoding:

### 1. Structured Code Generation

Constrained decoding can be used to generate structured code snippets. By setting appropriate constraints, we can ensure that the model produces well-formatted, syntactically correct code that includes specific elements we require. Here's an example:

```python
from phi_3_vision_mlx import generate, constrain

constrain(
    prompt="Write a Python function to calculate the Fibonacci sequence up to a given number n.", 
    constr=[
        (100, "\n```python\n"), 
        (100, " return "), 
        (200, "\n```")
    ], 
    use_beam=True
)
```

In this example, we're using constrained decoding to generate a Python function for calculating the Fibonacci sequence. The constraints serve several purposes:

1. The first constraint `(100, "\n```python\n")` ensures that the output begins with a Python code block, improving readability and making the code easy to copy and use.
2. The second constraint `(100, " return ")` guarantees that the function includes a return statement.
3. The final constraint `(200, "\n```")` closes the code block, ensuring a clean and complete code snippet.

These constraints help structure our generated code for clarity and functionality. This approach is particularly powerful for tasks that require consistent output formats or adherence to specific patterns. 

It's especially useful in the context of AI agents and function calling. Constrained decoding ensures AI agents produce properly formatted JSON responses, SQL queries with required clauses, or standardized API calls. It's equally valuable for more complex scenarios, like implementing specific design patterns or creating CRUD operation boilerplate.

In multi-agent systems, constrained decoding maintains consistent interfaces between components, allowing outputs from one model to serve reliably as inputs for another. This consistency is key for building robust, multi-step AI workflows and seamlessly integrating AI-generated code into larger systems.

## 2. Guided Reasoning in Complex Decision-Making

Constrained decoding can also guide the model's reasoning process in complex scenarios like medical diagnosis. Let's look at an example:

```python
prompts = [
    "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
    "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"
]

generate(prompts, blind_model=True)
# Output from the 1st prompt:
# D: Protein C deficiency\n\nProtein C deficiency is a rare genetic disorder that increases the risk of developing abnormal blood clots. It is an autosomal dominant disorder that can lead to venous thromboembolism. The patient's prolonged PTT and history of menorrhagia and easy bruising are consistent with a bleeding disorder. Protein C is a natural anticoagulant that inhibits the coagulation cascade. Deficiency of protein C can lead to a hypercoagulable state, which can result in venous thrombosis. The other options listed are not consistent with the patient's presentation. Factor V Leiden is a genetic disorder that increases the risk of developing abnormal blood clots, but it is associated with a prolonged PT and normal PTT. Hemophilia A is an X-linked recessive disorder that results in a deficiency of factor VIII, leading to a prolonged PTT. Lupus anticoagulant is an autoimmune disorder that can cause a prolonged PTT and is associated with antiphospholipid syndrome. Von Willebrand disease is a bleeding disorder caused by a deficiency or dysfunction of von Willebrand factor, which leads to a prolonged PTT and a history of menorrhagia. However, the patient's PTT is significantly prolonged, which is more consistent with protein C deficiency.<|end|>
# Output from the 2nd prompt:
# E: 24-hour urine protein\n\nThe patient presents with hypertension, proteinuria, and edema, which are the three diagnostic criteria for preeclampsia. The diagnosis of preeclampsia is made when a pregnant woman has hypertension and either proteinuria or end-organ dysfunction after 20 weeks of gestation. In this case, the patient has hypertension (blood pressure of 148/90 mm Hg on two separate occasions), edema, and a fetal heart rate of 151/min, which is slightly elevated but not indicative of fetal distress. The most appropriate test to confirm the diagnosis of preeclampsia is a 24-hour urine protein assessment. This test will help determine the degree of proteinuria, which is a key feature of preeclampsia. Other tests, such as bilirubin assessment, coagulation studies, hematocrit assessment, and leukocyte count with differential, are not specific for preeclampsia and would not be the first-line tests to confirm the diagnosis.<|end|>
```

In this unconstrained version, our model kind of jumps the gun, blurting out "D: Protein C deficiency" right off the bat, without really thinking it through. Once it's said that, it's stuck - even though it later correctly explains that Protein C deficiency actually causes blood clots, not bleeding. The model even mentions Von Willebrand disease, which would fit the symptoms better, but it can't backtrack. Instead, it doubles down and starts making up reasons to support its first guess. This is a classic case of an autoregressive model getting tunnel vision because of how it generates text one bit at a time.

This is where constrained decoding can help. By making the model "show its work" before giving an answer, we can help it come to more accurate conclusions. It's like asking a student to explain their reasoning before they circle their final answer. Let's see what happens when we use constrained decoding to guide our model's response:

```python
constrain(
    prompts=prompts,
    constrs=[
        (0, '\nThe'),
        (100, ' The correct answer is'),
        (1, 'X.')
    ],
    blind_model=True,
    use_beam=True
)
# Output from the 1st prompt:
# The patient's symptoms of menorrhagia and easy bruising, along with a prolonged partial thromboplastin time (PTT) and normal platelet count and prothrombin time (PT), are suggestive of a coagulation factor deficiency. The correct answer is E.
# Output from the 2nd prompt:
# The patient’s history and physical examination are consistent with preeclampsia. Preeclampsia is a multisystem disorder that is characterized by new-onset hypertension and proteinuria after 20 weeks of gestation. The patient’s blood pressure is elevated (>140/90 mm Hg) and she has 2+ pitting edema of the lower extremity. The correct answer is E.
```

In this constrained version, we've applied a structure that forces the model to provide reasoning before giving an answer, similar to a chain-of-thought approach. This approach offers several benefits:

1. **Forced Reasoning**: The model must explain its thought process before concluding, encouraging a more thorough analysis of the provided information.
2. **Transparency**: The reasoning is clearly laid out, allowing us to understand how the model arrived at its conclusion.
3. **Reduced Bias**: By delaying the answer until after the reasoning, we potentially reduce the risk of the model retrofitting its explanation to match a premature conclusion.
4. **Improved Accuracy**: As seen in the first prompt, the constrained approach led to the correct diagnosis (E: Von Willebrand disease) compared to the incorrect diagnosis in the unconstrained version.

This method of constrained decoding is analogous to asking a student to "show their work" in an exam. It not only provides the final answer but also gives insight into the thought process, making the output more informative, transparent, and potentially more accurate.

## Conclusion

By implementing constrained decoding in complex decision-making scenarios, we can create more reliable and interpretable AI systems. This is important in high-stakes domains like medical diagnosis, legal reasoning, or financial analysis, where understanding the reasoning behind a decision is as important as the decision itself.

In the next part of our series, we'll explore techniques for fine-tuning our model on custom datasets, allowing us to adapt Phi-3-Vision for specific tasks or domains.
