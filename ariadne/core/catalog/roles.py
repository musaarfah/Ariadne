"""Mapping TMDB's job strings onto the roles the model uses.

All credits are stored raw (D26), so this mapping is applied at read time and role scope stays
a configuration choice rather than a schema commitment.

The whitelists are exact and deliberately narrow. TMDB's Editing department on *The Godfather*
holds two `Editor`, five `Assistant Editor` and two `Additional Editor`; only the first is the
person whose choices shape the cut. Matching loosely on "editor" would put five assistants into
the taste model and dilute the effect being measured.

Department is checked as well as job, so a stray `Music` credit filed under Production cannot
be read as a composer.
"""

from collections.abc import Iterable
from dataclasses import dataclass

DIRECTOR = "director"
EDITOR = "editor"
CINEMATOGRAPHER = "cinematographer"
COMPOSER = "composer"
PRODUCTION_DESIGNER = "production_designer"
WRITER = "writer"

# Added in the expanded scope, pre-registered in research/preregistration-roles.md on one
# criterion: does this person make decisions that shape what the finished film is like?
CASTING = "casting"
PLAYBACK_SINGER = "playback_singer"
SOUND_EDITOR = "supervising_sound_editor"
SOUND_DESIGNER = "sound_designer"
COSTUME_DESIGNER = "costume_designer"
CHOREOGRAPHER = "choreographer"

# Cast. Excluded from the research scope on purpose — the thesis is about people a viewer *cannot*
# name — and included in the product scope, because an actor bar is what gives the crew bar its
# context (D98). TMDB returns cast without department or job, so both are synthesised at ingest.
ACTOR = "actor"


@dataclass(frozen=True)
class RoleSpec:
    role: str
    department: str
    jobs: frozenset[str]


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(DIRECTOR, "Directing", frozenset({"Director"})),
    RoleSpec(EDITOR, "Editing", frozenset({"Editor"})),
    RoleSpec(
        CINEMATOGRAPHER,
        "Camera",
        frozenset({"Director of Photography", "Cinematography"}),
    ),
    # TMDB is inconsistent here: "Original Music Composer" is canonical but older and
    # non-English entries often carry "Music" or "Composer" instead.
    RoleSpec(COMPOSER, "Sound", frozenset({"Original Music Composer", "Music", "Composer"})),
    RoleSpec(
        PRODUCTION_DESIGNER,
        "Art",
        frozenset({"Production Design", "Production Designer"}),
    ),
    # "Novel", "Book" and "Story" are excluded: they credit the source material's author, who
    # did not work on the film. A novelist is not a below-the-line collaborator.
    RoleSpec(WRITER, "Writing", frozenset({"Screenplay", "Writer", "Screenwriter"})),
    # --- the expanded scope ---
    RoleSpec(CASTING, "Production", frozenset({"Casting"})),
    # In Indian cinema the playback singer is a primary creative presence, not a technician, and
    # 308 of this library's films are Indian.
    RoleSpec(PLAYBACK_SINGER, "Sound", frozenset({"Playback Singer"})),
    RoleSpec(SOUND_EDITOR, "Sound", frozenset({"Supervising Sound Editor"})),
    RoleSpec(SOUND_DESIGNER, "Sound", frozenset({"Sound Designer"})),
    RoleSpec(COSTUME_DESIGNER, "Costume & Make-Up", frozenset({"Costume Design"})),
    RoleSpec(CHOREOGRAPHER, "Crew", frozenset({"Choreographer"})),
    # --- cast, product scope only ---
    RoleSpec(ACTOR, "Acting", frozenset({"Actor"})),
)

# Deliberately absent: Crew/Stunts is the largest available pool (86 people at 12+ films, 4,931
# individuals) and is excluded because a stunt performer executes rather than decides. Likewise
# re-recording mixers, foley artists, producers and executive producers. See the pre-registration.

# The original six-role scope, kept so before-and-after comparisons remain possible.
CORE_BELOW_THE_LINE = (EDITOR, CINEMATOGRAPHER, COMPOSER, PRODUCTION_DESIGNER, WRITER)

EXPANDED_BELOW_THE_LINE = (
    *CORE_BELOW_THE_LINE,
    CASTING,
    PLAYBACK_SINGER,
    SOUND_EDITOR,
    SOUND_DESIGNER,
    COSTUME_DESIGNER,
    CHOREOGRAPHER,
)

# Roles the model uses, in the order they are reported.
#
# BELOW_THE_LINE and ALL_ROLES are the *research* scope and must not gain actors: the Stage 1 result
# compares below-the-line crew against directors, and quietly adding cast would invalidate the
# comparison the writeup reports.
BELOW_THE_LINE = EXPANDED_BELOW_THE_LINE
ALL_ROLES = (DIRECTOR, *BELOW_THE_LINE)

CAST_ROLES = (ACTOR,)

# The *product* scope: everything a person can be credited as. Used by the decomposition and the
# full-feature recommender, never by the thesis models (D97).
PRODUCT_ROLES = (*CAST_ROLES, DIRECTOR, *BELOW_THE_LINE)

_BY_KEY = {(spec.department, job): spec.role for spec in ROLE_SPECS for job in spec.jobs}


def canonical_role(department: str, job: str) -> str | None:
    """The canonical role for a TMDB credit, or None if it is not one we model."""
    return _BY_KEY.get((department, job))


def roles_present(credits: Iterable[tuple[str, str]]) -> set[str]:
    """Which modelled roles a film's (department, job) pairs cover."""
    found = set()
    for department, job in credits:
        role = canonical_role(department, job)
        if role is not None:
            found.add(role)
    return found
