# This file was authored by GitHub user @couillonnade (PR #2)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from phi_3_vision_mlx import Agent

# Multi-turn VQA
agent = Agent()
agent('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
agent('What is the location?')
agent.end()

# Generative Feedback Loop
agent('Plot a Lissajous Curve.')
agent('Modify the code to plot 3:4 frequency')
agent.end()

# API Tool Use
agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
agent.end()
agent('Speak "People say nothing is impossible, but I do nothing every day."')
agent.end()

from phi_3_vision_mlx import constrain

# Code Generation
constrain ("Write a Python function to calculate the Fibonacci sequence up to a given number n.", [(100, "\n```python\n"), (100, " return "), (200, "\n```")])

# Multiple Choice Question Answering
prompts = [
    "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms? A: Factor V Leiden B: Hemophilia A C: Lupus anticoagulant D: Protein C deficiency E: Von Willebrand disease",
    "A 25-year-old primigravida presents to her physician for a routine prenatal visit. She is at 34 weeks gestation, as confirmed by an ultrasound examination. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore. The course of her pregnancy has been uneventful and she has been compliant with the recommended prenatal care. Her medical history is unremarkable. She has a 15-pound weight gain since the last visit 3 weeks ago. Her vital signs are as follows: blood pressure, 148/90 mm Hg; heart rate, 88/min; respiratory rate, 16/min; and temperature, 36.6℃ (97.9℉). The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg. The fetal heart rate is 151/min. The physical examination is significant for 2+ pitting edema of the lower extremity. Which of the following tests o should confirm the probable condition of this patient? A: Bilirubin assessment B: Coagulation studies C: Hematocrit assessment D: Leukocyte count with differential E: 24-hour urine protein"]
constrain(prompts, constraints=[(30, ' The correct answer is'), (10, 'X.')], blind_model=True, quantize_model=True)

from phi_3_vision_mlx import train_lora, test_lora

# Train
train_lora(lora_layers=5, lora_rank=16, epochs=10, take=10, batch_size=2, lr=1e-4, warmup=.5, dataset_path="JosefAlbers/akemiH_MedQA_Reason")

# Test
test_lora()

from phi_3_vision_mlx import benchmark

# Benchmark
benchmark()
