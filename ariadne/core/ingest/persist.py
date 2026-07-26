import secrets

from sqlalchemy.orm import Session

from ariadne.core.ingest.export import ParsedExport
from ariadne.db.models import DiaryEntry, Like, Rating, Upload, UploadStatus

# 32 bytes of urlsafe randomness is 43 characters, which fits Upload.token. With no accounts
# this token is the only handle on a result, so it has to be unguessable.
TOKEN_BYTES = 32


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def persist_export(session: Session, parsed: ParsedExport, token: str | None = None) -> Upload:
    """Write a parsed export to the database and return its Upload.

    Nothing derived from profile.csv reaches this function — ParsedExport has no field for
    it, because ExportSource never opened the file.
    """
    upload = Upload(
        token=token or new_token(),
        status=UploadStatus.PENDING,
        film_count=parsed.stats.ratings,
    )
    session.add(upload)
    session.flush()

    session.add_all(
        Rating(
            upload_id=upload.id,
            letterboxd_uri=rating.letterboxd_uri,
            rating=rating.rating,
            logged_date=rating.logged_date,
        )
        for rating in parsed.ratings
    )
    session.add_all(
        DiaryEntry(
            upload_id=upload.id,
            film_key=entry.key,
            watched_date=entry.watched_date,
            is_rewatch=entry.is_rewatch,
        )
        for entry in parsed.diary
    )
    session.add_all(
        Like(upload_id=upload.id, letterboxd_uri=like.letterboxd_uri) for like in parsed.likes
    )
    session.flush()

    return upload
