import re
from typing import List, Tuple, Pattern, Callable
import json
import os

Formatter = Callable[[re.Match], List[str]]

def split_comparison_question(question: str) -> List[str]:
    """
    Tách một câu hỏi so sánh phức tạp thành hai câu hỏi con dựa trên một bộ quy tắc Regex.
    """
    question = re.sub(r'\s+', ' ', question).strip()
    
    rules: List[Tuple[Pattern, Formatter]] = [
        (
            re.compile(r"Do both films (?P<entity1>.+?) and (?P<entity2>.+?) have the producers from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the producer of film {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of the producer of film {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Do both films:\s*(?P<entity1>.+?)\s+and\s+(?P<entity2>.+?)\s+have the producers from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the producer of film {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of the producer of film {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Which film has the director who is older than the other, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the director of film {m.group('entity1').strip()} born?",
                f"When was the director of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Which film (?:whose director was|has the director(?: who was| who)?|has the director) (?P<action>born|died) first, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When did the director of film {m.group('entity1').strip()} die?" if m.group('action') == 'died' else f"When was the director of film {m.group('entity1').strip()} born?",
                f"When did the director of film {m.group('entity2').strip()} die?" if m.group('action') == 'died' else f"When was the director of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Do both (?P<entity1>.+?) and (?P<entity2>.+?) films have the directors from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the director of film {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of the director of film {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Which film has the director (?:who )?((?:was )?born|died) (?P<attribute>later|earlier), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the director of film {m.group('entity1').strip()} {m.group(2).replace('was ', '')}?",
                f"When was the director of film {m.group('entity2').strip()} {m.group(2).replace('was ', '')}?"
            ]
        ),
        (
            re.compile(r"Which film whose director is (?P<attribute>younger|older), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the director of film {m.group('entity1').strip()} born?",
                f"When was the director of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Which film (came out first|was released more recently), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the film {m.group('entity1').strip()} released?",
                f"When was the film {m.group('entity2').strip()} released?"
            ]
        ),
        (
            re.compile(r"Do both films(?:[:,])? (?P<entity1>.+?) and (?P<entity2>.+) have the directors (?:who are )?from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the director of film {m.group('entity1').strip(' ,')}?",
                f"What is the {m.group('property')} of the director of film {m.group('entity2').strip(' ,')}?"
            ]
        ),
        (
            re.compile(r"(?:Do|Did) the movies (?P<entity1>.+?) and (?P<entity2>.+?), originate from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of origin of the movie {m.group('entity1').strip(' ,')}?",
                f"What is the {m.group('property')} of origin of the movie {m.group('entity2').strip(' ,')}?"
            ]
        ),
        (
            re.compile(r"Are director of film (?P<entity1>.+) and director of film (?P<entity2>.+) (?:both )?from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the director of film {m.group('entity1').strip()}?",
                f"What is the {m.group('property')} of the director of film {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Are the directors of films (?P<entity1>.+) and (?P<entity2>.+) (?:both )?from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the director of film {m.group('entity1').strip()}?",
                f"What is the {m.group('property')} of the director of film {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Are (?:both )?(?P<entity1>.+) and (?P<entity2>.+) (?:both )?located in the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"In which {m.group('property')} is {m.group('entity1').strip(' ,')} located?",
                f"In which {m.group('property')} is {m.group('entity2').strip(' ,')} located?"
            ]
        ),
        (
            re.compile(r"Does (?P<entity1>.+) have the same (?P<property>\w+) as (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of {m.group('entity1').strip()}?",
                f"What is the {m.group('property')} of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Do (?P<entity1>.+) and (?P<entity2>.+) (?:have|share) the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of {m.group('entity1').strip()}?",
                f"What is the {m.group('property')} of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Are (?P<entity1>.+) and (?P<entity2>.+) from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of {m.group('entity1').strip()}?",
                f"What is the {m.group('property')} of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"(?:Did|Are|Were) (?P<entity1>.+?) and (?P<entity2>.+?) (?:have|of|from) the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Who (is younger|died later), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} born?" if 'younger' in m.group(1) else f"When did {m.group('entity1').strip()} die?",
                f"When was {m.group('entity2').strip()} born?" if 'younger' in m.group(1) else f"When did {m.group('entity2').strip()} die?"
            ]
        ),
        (
            re.compile(r"Which film was released earlier, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the film {m.group('entity1').strip()} released?",
                f"When was the film {m.group('entity2').strip()} released?"
            ]
        ),
        (
            re.compile(r"Who was born first out of (?P<entity1>.+) and (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} born?",
                f"When was {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Was (?P<entity1>.+) or (?P<entity2>.+) born first\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} born?",
                f"When was {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Who (?:was born|died) (first|earlier|later), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} born?" if 'born' in m.group(1) else f"When did {m.group('entity1').strip()} die?",
                f"When was {m.group('entity2').strip()} born?" if 'born' in m.group(1) else f"When did {m.group('entity2').strip()} die?"
            ]
        ),
        (
            re.compile(r"Who is older, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} born?",
                f"When was {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Who lived longer, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"What were the birth and death dates of {m.group('entity1').strip()}?",
                f"What were the birth and death dates of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Which (?P<type>\w+) (?:was released|came out) (?:earlier|first), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the {m.group('type')} {m.group('entity1').strip()} released?",
                f"When was the {m.group('type')} {m.group('entity2').strip()} released?"
            ]
        ),
        (
            re.compile(r"Which film has the director who is older, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the director of film {m.group('entity1').strip()} born?",
                f"When was the director of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Which film has the producer born first, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the producer of film {m.group('entity1').strip()} born?",
                f"When was the producer of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Were both (?P<entity1>.+?) and (?P<entity2>.+?), born in the same place\?", re.IGNORECASE),
            lambda m: [
                f"Where was {m.group('entity1').strip()} born?",
                f"Where was {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Did (?P<entity1>.+?) and (?P<entity2>.+?) share the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Who has a wider scope of profession, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the profession of {m.group('entity1').strip()}?",
                f"What is the profession of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Which one was established first, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was {m.group('entity1').strip()} established?",
                f"When was {m.group('entity2').strip()} established?"
            ]
        ),
        (
            re.compile(r"Do the bands (?P<entity1>.+?) and (?P<entity2>.+?), originate from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of origin of the band {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of origin of the band {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Which magazine was founded first, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the magazine {m.group('entity1').strip()} founded?",
                f"When was the magazine {m.group('entity2').strip()} founded?"
            ]
        ),
        (
            re.compile(r"Which film has the producer who was born later, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the producer of film {m.group('entity1').strip()} born?",
                f"When was the producer of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Which album was released more recently, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the album {m.group('entity1').strip()} released?",
                f"When was the album {m.group('entity2').strip()} released?"
            ]
        ),
        (
            re.compile(r"Which film has more directors, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"How many directors does the film {m.group('entity1').strip()} have?",
                f"How many directors does the film {m.group('entity2').strip()} have?"
            ]
        ),
        (
            re.compile(r"Who has more scope of profession, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the scope of profession of {m.group('entity1').strip()}?",
                f"What is the scope of profession of {m.group('entity2').strip()}?"
            ]
        ),
        (
            re.compile(r"Which film has the producer (?:who was|who is|was)?\s*(?P<action>born|died) (?P<comparison>first|later|earlier), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the producer of film {m.group('entity1').strip()} {m.group('action')}?",
                f"When was the producer of film {m.group('entity2').strip()} {m.group('action')}?"
            ]
        ),
        (
            re.compile(r"Which film (?:has the producer who is|whose producer is) (?P<comparison>older|younger), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the producer of film {m.group('entity1').strip()} born?",
                f"When was the producer of film {m.group('entity2').strip()} born?"
            ]
        ),
        (
            re.compile(r"Do both (?:films:)?\s*(?P<entity1>.+?)\s+and\s+(?P<entity2>.+?)\s+films have the producers from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of the producer of film {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of the producer of film {m.group('entity2').strip()}"
            ]
        ),
        (
            re.compile(r"Which (?P<type>magazine|high school|band|museum|school) was (?:established|published|founded|formed|opened) (?:first|earlier), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the {m.group('type')} {m.group('entity1').strip()} established?",
                f"When was the {m.group('type')} {m.group('entity2').strip()} established?"
            ]
        ),
        (
            re.compile(r"Which film has more producers, (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"How many producers does the film {m.group('entity1').strip()} have?",
                f"How many producers does the film {m.group('entity2').strip()} have?"
            ]
        ),
        (
            re.compile(r"Which film whose producer was (?P<action>born|died) (?P<comparison>first|later|earlier), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the producer of film {m.group('entity1').strip()} {m.group('action')}?",
                f"When was the producer of film {m.group('entity2').strip()} {m.group('action')}?"
            ]
        ),
        (
            re.compile(r"Which (?P<type>university|song) was (?:founded|released) (?:first|more recently), (?P<entity1>.+) or (?P<entity2>.+)\?", re.IGNORECASE),
            lambda m: [
                f"When was the {m.group('type')} {m.group('entity1').strip()} {m.group(2)}?",
                f"When was the {m.group('type')} {m.group('entity2').strip()} {m.group(2)}?"
            ]
        ),
        (
            re.compile(r"Do the magazines (?P<entity1>.+?) and (?P<entity2>.+?), originate from the same (?P<property>\w+)\?", re.IGNORECASE),
            lambda m: [
                f"What is the {m.group('property')} of origin of the magazine {m.group('entity1').strip()}",
                f"What is the {m.group('property')} of origin of the magazine {m.group('entity2').strip()}"
            ]
        ),
    ]

    for pattern, formatter in rules:
        match = pattern.match(question)
        if match:
            try:
                result = formatter(match)
                if isinstance(result, list) and len(result) == 2:
                    return result
            except IndexError:
                continue 

    return [question]
