# P5: Plug-and-Play Persona Prompting for Personalized Response Selection

## Requirements
1. Pytorch 1.8
2. Python 3.6
3. Transformer 4.4.0
4. datasets
5. scipy

## [Dataset](http://naver.me/xHnBDXO3)
- PERSONA-CHAT: positive 1, negative 19 (given)
- Focus: positive 1 (given), context negative sampling 2, random negative sampling 17

#### Dataset folder location
- ./dataset/FoCus
- ./dataset/personachat

## Pretrained Model
- [PersonaChat model]()
- [Focus model]()

#### Model folder location
- ./model/NP_focus/roberta-base/model.bin
- ./model/NP_persona/roberta-base/model.bin
- ./model/NP_persona/roberta-large/model.bin
- ./model/prompt_finetuning/roberta-{size}/{personachat_type}/model.bin

## NP_focus (NP: No Persona)
**standard response selection model for Focus**

```bash
cd NP_focus
python3 train.py --model_type roberta-{size} --epoch 10
```
- size: base or large

## NP_persona
**standard response selection model for PERSONA-CHAT**

```bash
cd NP_focus
python3 train.py --model_type roberta-{size} --data_type {data_type} --epoch 10
```
- size: base or large
- data_type: original or revised

## SoP (Similarity of Persona)
**Zero-shot baseline**
- test_focus.py: testing for focus dataset
- test_perchat.py: testing for personachat dataset

```bash
cd SoP
python3 test_perchat.py --model_type roberta-{size} --data_type {data_type} --persona simcse --weight {weight} --agg max
```
- size: base or large
- data_type: original or revised (It doesn't matter in focus))
- weight
    - 0.5 for original persona and Focus
    - 0.05 for revised persona

## prompt_finetuning
**Fine-tuned P5 model**
- train.py: The main training file mentioned in the paper
- train_no_ground: don't use persona grounding
- train_no_question: don't use prompt question

```bash
cd prompt_finetuning
python3 train.py --model_type roberta-{size} --data_type {data_type} --persona_type {persona_type} --persona {persona} --num_of_persona {num_of_persona} --reverse
```
- size: base or large
- data_type: personachat or focus
- persona_type: original or revised (It doesn't matter in focus)
- persona: simcse or nli or bertscore (recommend: simcse)
- num_of_persona: 1 to 5  (recommend: 2)
- reverse: option (order of persona sentences)

## prompt_persona_context
**Zero-shot P5 model** (ablation study)
- test.py: The main test file mentioned in the paper
- test_no_ground.py: don't use persona grounding
- test_no_question.py: don't use prompt question
- test_other_question.py: variant of prompt question
- test_random_question.py: random prompt question
- *dd* means: When SRS is trained with dailydailog

```bash
cd prompt_persona_context
python3 test.py --model_type roberta-{size} --data_type {data_type} --persona_type {persona_type} --persona {persona} --num_of_persona {num_of_persona} --reverse
```
- size: base or large
- data_type: personachat or focus
- persona_type: original or revised (It doesn't matter in focus)
- persona: simcse or nli or bertscore (recommend: simcse)
- num_of_persona: 1 to 5  (recommend: 2)
- reverse: option (order of persona sentences)
