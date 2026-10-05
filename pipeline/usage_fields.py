"""How each project calls the library, reduced to what one point on a map can show.

Stage 05 records every call a project makes to the library's classes. A point has one
colour per colormap and a project can make many calls, with different arguments, so each
function here has a rule for turning several calls into one answer. What a rule leaves
out stays on the hovercard and in the search text.
"""

import json
import re
from collections import Counter

from code_usage import EXPRESSION
from config import Feature

# The kinds of answer `value_given` returns.
VALUE, SEVERAL, DEFAULT, UNREADABLE = "value", "several", "default", "unreadable"
# The other answers `how_often_given` returns.
EVERY, SOME, NEVER = "every", "some", "never"

# A value longer than this, in the search text, is somebody's path or sentence.
MAX_TOKEN_VALUE = 40


def constructor_calls(calls) -> list[dict]:
    """One project's `calls` from stage 05: its distinct constructor calls, commonest first."""
    return json.loads(calls) if isinstance(calls, str) else []


def features_used(
    features: tuple[Feature, ...],
    called: list[str],
    methods: list[str],
    fits_with_target: bool,
    calls: list[dict],
) -> list[str]:
    """The label of every feature a project's code shows, in the order config lists them."""

    def holds(feature: Feature) -> bool:
        if feature.method and feature.method in methods:
            return True
        if feature.called and any(re.search(feature.called, name) for name in called):
            return True
        if feature.fit_with_target and fits_with_target:
            return True
        if feature.argument:
            key, wanted = feature.argument
            given = [call["kwargs"][key] for call in calls if key in call["kwargs"]]
            # The type as well: True is equal to 1, and a count of 1 is not a switch.
            return any(type(value) is type(wanted) and value == wanted for value in given)
        return False

    return [feature.label for feature in features if holds(feature)]


def rarest_feature(used: list[list[str]], features: tuple[Feature, ...]) -> list[str | None]:
    """One feature per project: of those it uses, the one the fewest projects use.

    Nearly every project with two features pairs a common one with a rare one. Showing
    the rare one keeps each rare feature's projects together under its colour, and takes
    them from the common one, which has projects to spare. Ties go to config's order.
    """
    counts = Counter(label for labels in used for label in labels)
    rank = {feature.label: (counts[feature.label], i) for i, feature in enumerate(features)}
    return [min(labels, key=rank.__getitem__) if labels else None for labels in used]


def value_given(calls: list[dict], parameter: str) -> tuple[str, object]:
    """What a project's calls give one constructor argument, as one answer.

    (VALUE, v) when v is the only value it gives, in the calls that give one that can be
    read. A call that leaves the argument out does not make a second answer: the answer
    is what the project sets it to when it sets it. (SEVERAL, None) for two or more
    values, (DEFAULT, None) when its calls leave the argument out, and (UNREADABLE, None)
    when it is given something that is not written out, or nothing can be told.
    """
    values, hidden, left_out = set(), False, False
    for call in calls:
        if parameter in call["kwargs"]:
            value = call["kwargs"][parameter]
            if value == EXPRESSION:
                hidden = True
            else:
                # As JSON so that 2, 2.0 and "2" stay apart.
                values.add(json.dumps(value))
        elif not call["star"]:
            left_out = True
        # Otherwise the call passes arguments through **, and this may be one of them.
    if len(values) > 1:
        return SEVERAL, None
    if values:
        return VALUE, json.loads(values.pop())
    if left_out and not hidden:
        return DEFAULT, None
    return UNREADABLE, None


def how_often_given(calls: list[dict], parameter: str) -> str:
    """Whether a project's calls give an argument a value: in EVERY call, SOME, or NEVER.

    Counted over the calls whose arguments can be read; UNREADABLE when none can. Passing
    None is not giving a value. A value that is not written out still is one.
    """
    given = left_out = 0
    for call in calls:
        if parameter in call["kwargs"]:
            if call["kwargs"][parameter] is None:
                left_out += 1
            else:
                given += 1
        elif not call["star"]:
            left_out += 1
    if not given and not left_out:
        return UNREADABLE
    if not left_out:
        return EVERY
    return SOME if given else NEVER


def argument_tokens(calls: list[dict]) -> list[str]:
    """`name=value` for every argument value a project's calls write out, for the search text."""
    tokens = set()
    for call in calls:
        for key, value in call["kwargs"].items():
            if value != EXPRESSION and len(str(value)) <= MAX_TOKEN_VALUE:
                tokens.add(f"{key}={value}")
    return sorted(tokens)
