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
)

# Roles the model uses, in the order they are reported.
BELOW_THE_LINE = (EDITOR, CINEMATOGRAPHER, COMPOSER, PRODUCTION_DESIGNER, WRITER)
ALL_ROLES = (DIRECTOR, *BELOW_THE_LINE)

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
