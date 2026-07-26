"""Role mapping.

The narrowness is the point. TMDB's Editing department on The Godfather holds two `Editor`,
five `Assistant Editor` and two `Additional Editor`; only the first shapes the cut. Matching
loosely on "editor" would put five assistants into the taste model.
"""

import pytest

from ariadne.core.catalog.roles import (
    ALL_ROLES,
    BELOW_THE_LINE,
    CINEMATOGRAPHER,
    COMPOSER,
    DIRECTOR,
    EDITOR,
    PRODUCTION_DESIGNER,
    ROLE_SPECS,
    WRITER,
    canonical_role,
    roles_present,
)


@pytest.mark.parametrize(
    ("department", "job", "expected"),
    [
        ("Directing", "Director", DIRECTOR),
        ("Editing", "Editor", EDITOR),
        ("Camera", "Director of Photography", CINEMATOGRAPHER),
        ("Camera", "Cinematography", CINEMATOGRAPHER),
        ("Sound", "Original Music Composer", COMPOSER),
        ("Sound", "Music", COMPOSER),
        ("Art", "Production Design", PRODUCTION_DESIGNER),
        ("Writing", "Screenplay", WRITER),
        ("Writing", "Writer", WRITER),
    ],
)
def test_recognised_credits(department: str, job: str, expected: str):
    assert canonical_role(department, job) == expected


@pytest.mark.parametrize(
    ("department", "job"),
    [
        # Assistants and additional hands are not the person whose choices shape the film.
        ("Editing", "Assistant Editor"),
        ("Editing", "Additional Editor"),
        ("Directing", "Assistant Director"),
        ("Directing", "Script Supervisor"),
        ("Camera", "Camera Operator"),
        ("Camera", "Still Photographer"),
        ("Camera", "Assistant Camera"),
        ("Sound", "Musician"),
        ("Sound", "Conductor"),
        ("Sound", "Sound Re-Recording Mixer"),
        ("Art", "Art Direction"),
        ("Art", "Set Decoration"),
        ("Art", "Assistant Art Director"),
        # Source-material credits: the novelist did not work on the film.
        ("Writing", "Novel"),
        ("Writing", "Book"),
        ("Writing", "Story"),
        ("Writing", "Story Editor"),
        ("Production", "Producer"),
        ("Crew", "Stunts"),
        ("Lighting", "Gaffer"),
    ],
)
def test_unrecognised_credits(department: str, job: str):
    assert canonical_role(department, job) is None


def test_department_guards_the_job():
    """A "Music" credit filed outside Sound must not read as a composer."""
    assert canonical_role("Sound", "Music") == COMPOSER
    assert canonical_role("Production", "Music") is None
    assert canonical_role("Crew", "Editor") is None


def test_roles_present_collects_only_modelled_roles():
    credits = [
        ("Directing", "Director"),
        ("Directing", "Assistant Director"),
        ("Editing", "Editor"),
        ("Editing", "Assistant Editor"),
        ("Production", "Producer"),
    ]
    assert roles_present(credits) == {DIRECTOR, EDITOR}


def test_director_is_not_below_the_line():
    """The thesis compares below-the-line crew against the director, so they cannot mix."""
    assert DIRECTOR not in BELOW_THE_LINE
    assert DIRECTOR in ALL_ROLES
    assert set(BELOW_THE_LINE) < set(ALL_ROLES)


def test_every_spec_is_reachable():
    for spec in ROLE_SPECS:
        for job in spec.jobs:
            assert canonical_role(spec.department, job) == spec.role
